import os
import sys
import argparse
import copy
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert
import cv2
import matplotlib.pyplot as plt
import PIL
import requests
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from openai import OpenAI
from VLM_CaP.src.key import mykey
import time
import base64

# 设置环境变量
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 导入自定义模块
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict, load_image_from_array
from track_anything import TrackingAnything
from track_anything import parse_augment
import supervision as sv
from segment_anything import build_sam, SamPredictor

# 加载模型函数
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    try:
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
        args = SLConfig.fromfile(cache_config_file)
        model = build_model(args)
        args.device = device
        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location='cpu')
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print(f"Model loaded from {cache_file} \n => {log}")
        _ = model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

# 读取视频函数
def read_video(video_path):
    try:
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            print("Error: Could not open video.")
            sys.exit(1)
        frames = []
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(range(total_frames), desc="Reading video"):
            ret, frame = video_capture.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        video_capture.release()
        return frames
    except Exception as e:
        print(f"Error reading video: {e}")
        sys.exit(1)

# 写入视频函数
def write_video(frames, output_path, fps):
    if not frames:
        print("Error: No frames to write.")
        return
    try:
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in tqdm(frames, desc="Writing video"):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        print('Write video success')
        video_writer.release()
    except Exception as e:
        print(f"Error writing video: {e}")


def image_to_base64(image):
    """
    将PIL图像转换为Base64编码的字符串。
    :param image: 输入的PIL图像。
    :return: Base64编码的字符串。
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# def for calling openai api with different prompts
def call_openai_api(prompt_messages):
    params = {
        "model": "gpt-4o",
        "messages": prompt_messages,
        "max_tokens": 400,
        "temperature": 0
    }
    result = client.chat.completions.create(**params)
    return result.choices[0].message.content

def get_object_list(selected_frames):
    # first prompt to get objects in the environment
    encoded_image = image_to_base64(Image.fromarray(selected_frames[0]))
    prompt_messages_state = [
        {
            "role": "user",
            "content": [
                "There are two kinds of objects, grasped_objects and containers in the environment. We only care about objects on the desk.",
                "Based on the input picture, answer:",
                "1. How many objects are there in the environment?",
                "2. What are these objects?",
                "You should respond in the format of the following example:",
                "Number: 1",
                "Objects: red pepper, red tomato, white bowl",
                {"image": encoded_image, "resize": 768},
            ],
        },
    ]
    response_state = call_openai_api(prompt_messages_state)
    return response_state

# 主函数
def main():
    start_time = time.time()
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
    print(f"GroundingDINO model loaded in {time.time() - start_time:.2f} seconds")

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sam_checkpoint = 'sam_vit_h_4b8939.pth'
    start_time = time.time()
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    print(f"SAM model loaded in {time.time() - start_time:.2f} seconds")

    if DEVICE.type == 'cpu':
        float_type = torch.float32
    else:
        float_type = torch.float16

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=float_type,
    )
    if DEVICE.type != 'cpu':
        pipe = pipe.to("cuda")

    video_path = '/home/bw2716/VLMTutor/Cropped_real_world_demo.mp4'
    output_video = '/home/bw2716/VLMTutor/Cropped_real_world_demo_track.mp4'
    start_time = time.time()
    frames = read_video(video_path)
    print(f"Video read in {time.time() - start_time:.2f} seconds, total frames: {len(frames)}")

    object_list_response = get_object_list(frames[0:1])

    # 将对象列表转换为分号分隔的字符串
    TEXT_PROMPT = object_list_response
    print(f"Generated prompt: {TEXT_PROMPT}")

    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25

    image_source, image = load_image_from_array(frames[0])

    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device=DEVICE
    )

    def my_annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases) -> np.ndarray:
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
        for box, logit, phrase in zip(xyxy, logits, phrases):
            x1, y1, x2, y2 = map(int, box)
            label = f"{phrase} {logit:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - 4), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return annotated_frame

    annotated_frame = my_annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    plt.imshow(annotated_frame)
    plt.axis('off')
    plt.show()

    sam_predictor.set_image(image_source)
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(DEVICE)
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # def show_mask(mask, image, random_color=True):
    #     if random_color:
    #         color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    #     else:
    #         color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    #     h, w = mask.shape[-2:]
    #     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    #     annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    #     mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
    #     return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))
    def show_mask(mask, image, contour_color=(0, 255, 0), contour_thickness=2):
        """
        显示掩码轮廓而不填充内部颜色。
        :param mask: 掩码，PyTorch Tensor，形状为 (H, W)。
        :param image: 原始图像，形状为 (H, W, 3)。
        :param contour_color: 轮廓的颜色，默认为绿色。
        :param contour_thickness: 轮廓的厚度，默认为2。
        :return: 带有轮廓的图像，格式为 numpy 数组 (H, W, 3)。
        """
        # 将 PyTorch Tensor 转换为 numpy 数组
        mask_np = mask.cpu().numpy()
        # 将掩码转换为8位图像
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        # 查找轮廓
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 将图像从 numpy 数组转换为 PIL 图像格式
        annotated_frame_pil = Image.fromarray(image).convert("RGBA")
        # 创建一个透明图层用于绘制轮廓
        overlay = Image.new("RGBA", annotated_frame_pil.size)
        draw = ImageDraw.Draw(overlay)
        # 绘制轮廓
        for contour in contours:
            # 将轮廓点转换为列表并绘制
            contour_points = [(int(point[0][0]), int(point[0][1])) for point in contour]
            draw.line(contour_points, fill=contour_color + (255,), width=contour_thickness)
        # 叠加图像和轮廓
        annotated_frame_pil = Image.alpha_composite(annotated_frame_pil, overlay)
        # 转换回 RGB 格式的 numpy 数组
        annotated_frame = np.array(annotated_frame_pil.convert("RGB"))
        return annotated_frame

    masks = masks.cpu()
    annotated_frame_with_mask = show_mask(masks[0][0], annotated_frame)
    masks_np = masks.numpy()

    h, w = masks_np[0][0].shape
    pixel_cnt = h * w
    indices_to_keep = np.ones(len(masks_np), dtype=bool)
    for i in range(len(masks_np)):
        if np.sum(masks_np[i][0]) > pixel_cnt * 0.5:
            indices_to_keep[i] = False
    masks_np = masks_np[indices_to_keep]

    multi_mask = np.zeros(masks_np[0][0].shape)
    for i in range(len(masks_np)):
        multi_mask[masks_np[i][0]] = i + 1

    xmem_checkpoint = 'Track-Anything/checkpoints/XMem-s012.pth'
    e2fgvi_checkpoint = 'Track-Anything/checkpoints/E2FGVI-HQ-CVPR22.pth'
    args = parse_augment()
    model = TrackingAnything(sam_checkpoint, xmem_checkpoint, e2fgvi_checkpoint,args)
    masks, logits, painted_images = model.generator(images=frames, template_mask=multi_mask)

    write_video(painted_images, output_video, 30)

if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run video processing script")
#     parser.add_argument('--input_video', type=str, required=True, help='Path to the input video')
#     parser.add_argument('--output_video', type=str, required=True, help='Path to the output video')
#     args = parser.parse_args()
    client = OpenAI(api_key=mykey)
    main()