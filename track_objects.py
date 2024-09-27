import os
import sys
import argparse
import copy
import gc
import json
import re
import csv
import base64
from io import BytesIO
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert
from tqdm import tqdm
import cv2
import scipy.signal
import matplotlib.pyplot as plt

from openai import OpenAI
from VLM_CaP.src.key import mykey, projectkey
from diffusers import StableDiffusionInpaintPipeline
from sam2.build_sam import build_sam2_video_predictor

# Grounding DINO
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import (
    annotate,
    load_image,
    predict,
    load_image_from_array,
)

# Segment Anything
from segment_anything import build_sam, SamPredictor

# Hugging Face Hub
from huggingface_hub import hf_hub_download

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def call_openai_api(prompt_messages, client):
    params = {
        "model": "gpt-4o",
        "messages": prompt_messages,
        "max_tokens": 400,
        "temperature": 0,
    }
    result = client.chat.completions.create(**params)
    return result.choices[0].message.content


def get_object_list(video_path, client):
    # Use the first frame for encoding
    video = cv2.VideoCapture(video_path)

    base64Frames = []
    frame_count = 0
    max_frames = 2  # Only process the first 2 frames

    while video.isOpened() and frame_count < max_frames:
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        frame_count += 1

    video.release()
    print(len(base64Frames), "frames read.")

    prompt_messages_state = [
        {
            "role": "system",
            "content": [
                "You are a visual object detector. Your task is to count and identify the objects in the provided image that are on the desk. Focus on objects classified as grasped_objects and containers.",
                "Do not include hand or gripper in your answer",
            ],
        },
        {
            "role": "user",
            "content": [
                "There are two kinds of objects, grasped_objects and containers in the environment. We only care about objects on the desk.",
                "You must strictly follow the rules below: Even if there are multiple objects that appear identical, you must repeat their names in your answer according to their quantity. For example, if there are three wooden blocks, you must mention 'wooden block' three times in your answer."
                "Be careful and accurate with the number. Do not miss or add additional object in your answer."
                "Based on the input picture, answer:",
                "1. How many objects are there in the environment?",
                "2. What are these objects?",
                "You should respond in the format of the following example:",
                "Number: 3",
                "Objects: red pepper, red tomato, white bowl",
                "Number: 4",
                "Objects: wooden block, wooden block, wooden block, wooden block",
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0:1]),
            ],
        },
    ]

    response_state = call_openai_api(prompt_messages_state, client)
    return response_state


def extract_num_object(response_state):
    # Extract number of objects
    num_match = re.search(r"Number: (\d+)", response_state)
    num = int(num_match.group(1)) if num_match else 0

    # Extract objects
    objects_match = re.search(r"Objects: (.+)", response_state)
    objects_list = objects_match.group(1).split(", ") if objects_match else []

    # Construct object list
    objects = [obj for obj in objects_list]

    return num, objects


def load_model_hf(repo_id, filename, ckpt_config_filename, device="cpu"):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location="cpu")
    log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


def read_video(video_path):
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print("Error: Could not open video.")
        exit()

    frames = []

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    return frames


def my_annotate(
    image_source: np.ndarray,
    boxes: torch.Tensor,
    logits: torch.Tensor,
    phrases,
) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)

    for box, logit, phrase in zip(xyxy, logits, phrases):
        x1, y1, x2, y2 = map(int, box)
        label = f"{phrase} {logit:.2f}"

        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label background box
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            annotated_frame,
            (x1, y1 - text_height - 4),
            (x1 + text_width, y1),
            (0, 255, 0),
            -1,
        )

        # Draw label text
        cv2.putText(
            annotated_frame,
            label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    return annotated_frame


def video2jpg(video_path, output_folder, sample_freq=1):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        frame_index = 0
        save_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % sample_freq == 0:
                frame_filename = os.path.join(output_folder, f"{save_index:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                save_index += 1

            frame_index += 1

        cap.release()
        print(f"All frames have been saved to {output_folder}.")


color_list = {
    0: np.array([255, 0, 0]),       # Red
    1: np.array([0, 255, 0]),       # Green
    2: np.array([0, 0, 255]),       # Blue
    3: np.array([0, 125, 125]),     # Teal
    4: np.array([125, 0, 125]),     # Purple
    5: np.array([125, 125, 0]),     # Yellow
    6: np.array([255, 165, 0]),     # Orange
    7: np.array([255, 105, 180]),   # Pink
}


def contour_painter(
    input_image,
    input_mask,
    mask_color=5,
    mask_alpha=0.7,
    contour_color=1,
    contour_width=3,
    ann_obj_id=None,
):
    assert (
        input_image.shape[:2] == input_mask.shape
    ), "Different shape between image and mask"
    # 0: background, 1: foreground
    mask = np.clip(input_mask, 0, 1).astype(np.uint8)
    contour_radius = (contour_width - 1) // 2

    dist_transform_fore = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    dist_transform_back = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
    dist_map = dist_transform_fore - dist_transform_back
    contour_radius += 2
    contour_mask = np.abs(np.clip(dist_map, -contour_radius, contour_radius))
    contour_mask = contour_mask / np.max(contour_mask)
    contour_mask[contour_mask > 0.5] = 1.0

    # Paint contour
    painted_image = input_image.copy()
    color = color_list[contour_color]
    mask = 1 - contour_mask
    painted_image[mask.astype(bool)] = (
        painted_image[mask.astype(bool)] * (1 - 1) + color * 1
    ).astype("uint8")

    # Find the center position of the mask
    moments = cv2.moments(mask)
    if moments["m00"] != 0 and ann_obj_id is not None:
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_color = (0, 0, 0)
        font_thickness = 3
        cv2.putText(
            painted_image,
            str(ann_obj_id),
            (cX, cY),
            font,
            font_scale,
            font_color,
            font_thickness,
        )

    return painted_image


def write_video(frames, output_path, fps):
    if not frames:
        print("Error: No frames to write.")
        return

    height, width, _ = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    print("Video writing completed.")
    video_writer.release()


def process_mask_signal(mask_add, mask_min):
    kernel_size = 3

    n = len(mask_add)

    fig, axes = plt.subplots(n * 2, 1, figsize=(10, 5 * n * 2), sharex=True, sharey=True)
    axes = axes.flatten()

    filtered_mask_add = {}
    filtered_mask_min = {}
    min_num = 99999
    max_num = 0

    index = 0
    for k in mask_add.keys():
        mask1 = np.array(mask_add[k])
        mask2 = np.array(mask_min[k])

        # Median filter
        filtered_data1 = scipy.signal.medfilt(mask1, kernel_size=kernel_size)
        filtered_data2 = scipy.signal.medfilt(mask2, kernel_size=kernel_size)

        filtered_mask_add[k] = filtered_data1
        filtered_mask_min[k] = filtered_data2

        max_num = max(max_num, max(filtered_mask_add[k]))
        max_num = max(max_num, max(filtered_mask_min[k]))
        min_num = min(min_num, min(filtered_mask_add[k]))
        min_num = min(min_num, min(filtered_mask_min[k]))

        axes[index].plot(filtered_data1, linestyle="-", color="b")
        axes[index + 1].plot(filtered_data2, linestyle="-", color="b")
        index += 2

    plt.show()

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    fig, axes = plt.subplots(n * 2, 1, figsize=(10, 5 * n * 2), sharex=True, sharey=True)
    axes = axes.flatten()
    index = 0

    for k in filtered_mask_add.keys():
        filtered_mask_add[k] = ((filtered_mask_add[k] - min_num) / max_num) * 2 - 1
        filtered_mask_min[k] = ((filtered_mask_min[k] - min_num) / max_num) * 2 - 1

        filtered_mask_add[k] = sigmoid(filtered_mask_add[k] * 5)
        filtered_mask_min[k] = sigmoid(filtered_mask_min[k] * 5)

        axes[index].plot(filtered_mask_add[k], linestyle="-", color="b")
        axes[index + 1].plot(filtered_mask_min[k], linestyle="-", color="b")
        index += 2

    plt.show()

    fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n), sharex=True, sharey=True)
    axes = axes.flatten()
    index = 0

    final_result = {}
    for k in filtered_mask_add.keys():
        final_result[k] = filtered_mask_add[k] * filtered_mask_min[k]

        axes[index].plot(final_result[k], linestyle="-", color="b")
        index += 1

    plt.show()


def main(input_video_path, output_video_path, key_frames, bbx_file):
    # First Part: Get object list from first key_frame using VLM
    client = OpenAI(api_key=projectkey)

    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    groundingdino_model = load_model_hf(
        ckpt_repo_id, ckpt_filenmae, ckpt_config_filename
    )

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    if DEVICE.type == "cpu":
        float_type = torch.float32
    else:
        float_type = torch.float16

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=float_type,
    )

    if DEVICE.type != "cpu":
        pipe = pipe.to("cuda")

    video_path = input_video_path
    sample_freq = 16
    output_video_path = output_video_path

    frames = read_video(video_path)

    object_list_response = get_object_list(video_path, client)

    num, obj_list = extract_num_object(object_list_response)
    print(f"Generated prompt: {obj_list}")

    # Second Part: Use GroundedSAM2 to track the objects
    # Parameters for GroundingDINO
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25
    object_counts = Counter(obj_list)

    image_source, image = load_image_from_array(frames[0])

    best_boxes = []
    best_phrases = []
    best_logits = []

    # Iterate over each object and select the box with highest confidence
    for obj, count in object_counts.items():
        boxes, logits, phrases = predict(
            model=groundingdino_model,
            image=image,
            caption=obj,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=DEVICE,
        )

        if boxes.shape[0] > 0:
            selected_count = min(
                count, boxes.shape[0]
            )  # If returned boxes are fewer than object count
            for i in range(selected_count):
                best_boxes.append(boxes[i].unsqueeze(0))
                best_phrases.append(phrases[i])
                best_logits.append(logits[i])

    if best_boxes:
        best_boxes = torch.cat(best_boxes)
        best_logits = torch.stack(best_logits)

    annotated_frame = my_annotate(
        image_source=image_source,
        boxes=best_boxes,
        logits=best_logits,
        phrases=best_phrases,
    )
    annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB

    sam_predictor.set_image(image_source)
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(best_boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        boxes_xyxy, image_source.shape[:2]
    ).to(DEVICE)
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    masks = masks.cpu()
    masks_np = masks.numpy()

    h, w = masks_np[0][0].shape
    pixel_cnt = h * w
    indices_to_keep = np.ones(len(masks_np), dtype=bool)
    for i in range(len(masks_np)):
        if np.sum(masks_np[i][0]) > pixel_cnt * 0.3:
            indices_to_keep[i] = False
    masks_np = masks_np[indices_to_keep]

    del groundingdino_model
    del sam
    del sam_predictor
    del pipe

    torch.cuda.empty_cache()
    gc.collect()

    # Use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam2_checkpoint = "segment-anything-2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    predictor = build_sam2_video_predictor(
        model_cfg, sam2_checkpoint, device="cuda:0"
    )

    # First Round for sampling
    video_dir = (
        os.path.dirname(video_path)
        + f"/sample_freq_{sample_freq}_"
        + video_path.split("/")[-1].split(".")[0]
    )
    if not os.path.exists(video_dir):
        video2jpg(video_path, video_dir, sample_freq)

    frame_names = [
        p
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    prompts = {}  # Hold all the clicks we add for visualization

    ann_frame_idx = 0  # The frame index we interact with
    ann_obj_id = 1  # Give a unique id to each object we interact with

    for i in range(len(masks_np)):
        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=i,
            mask=masks_np[i][0],
        )

    # Run propagation throughout the video and collect the results in a dict
    video_segments = {}  # Contains the per-frame segmentation results
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Second Round for processing whole video
    del inference_state
    del predictor
    torch.cuda.empty_cache()
    torch.cuda.set_device(1)

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    video_dir = os.path.dirname(video_path) + "/" + video_path.split("/")[-1].split(".")[0]
    if not os.path.exists(video_dir):
        video2jpg(video_path, video_dir, 1)

    frame_names = [
        p
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    prompts = {}

    ann_frame_idx = 0
    ann_obj_id = 1

    for frame_idx in range(0, len(frame_names), sample_freq):
        for k in video_segments[frame_idx // sample_freq].keys():
            _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=k,
                mask=video_segments[frame_idx // sample_freq][k][0],
            )

    video_segments = {}
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Third Part: Select key frames and compute center coordinates of masks
    key_frame_coordinates = {}
    for frame_idx in key_frames:
        current_frame_coords = []

        if frame_idx in video_segments:
            for obj_id, mask in video_segments[frame_idx].items():
                mask_data = mask[0]
                mask_indices = np.argwhere(mask_data > 0)

                if len(mask_indices) > 0:
                    avg_y, avg_x = np.mean(mask_indices, axis=0)
                    current_frame_coords.append(
                        f"Object {obj_id}: ({int(avg_x)}, {int(avg_y)})"
                    )
                else:
                    print(
                        f"Warning: Empty mask for object {obj_id} in frame {frame_idx}"
                    )

        key_frame_coordinates[f"key_frame{frame_idx}"] = current_frame_coords

    for key_frame, coordinates in key_frame_coordinates.items():
        print(f"{key_frame}: {coordinates}")

    output_file = bbx_file

    # Store the coordinates in an output file
    with open(output_file, mode="a", newline="") as file:
        writer = csv.writer(file)

        all_coordinates = []
        for key_frame, coordinates in key_frame_coordinates.items():
            coordinates_str = "\n".join(coordinates)
            all_coordinates.append(f"{key_frame}: {coordinates_str}")

        final_coordinates_str = "\n".join(all_coordinates)

        writer.writerow([input_video_path, final_coordinates_str])

    # Fourth Part: Append all the painted frames into a video
    painted_frames = []
    for i in range(len(frame_names)):
        img = cv2.imread(os.path.join(video_dir, frame_names[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for k in video_segments[i].keys():
            img = contour_painter(
                img, video_segments[i][k][0], contour_color=1, ann_obj_id=k
            )
        painted_frames.append(img)

    mask_add = {}
    mask_min = {}
    for k in video_segments[i].keys():
        mask_add[k] = []
        mask_min[k] = []

    write_video(painted_frames, output_video_path, fps=30)

    for i in range(len(frame_names) - 1):
        for k in video_segments[i].keys():
            mask_before = video_segments[i][k][0].copy()
            mask_after = video_segments[i + 1][k][0].copy()
            mask_after[mask_before] = False
            add_cnt = np.sum(mask_after)

            mask_before = video_segments[i][k][0].copy()
            mask_after = video_segments[i + 1][k][0].copy()
            mask_before[mask_after] = False
            min_cnt = np.sum(mask_before)

            mask_add[k].append(add_cnt.item())
            mask_min[k].append(min_cnt.item())

    process_mask_signal(mask_add, mask_min)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process video with SAM and GroundingDINO."
    )
    parser.add_argument("--input", type=str, help="Path to the input video")
    parser.add_argument("--output", type=str, help="Path to the output video")
    parser.add_argument("--key_frames", nargs="+", type=int, help="List of key frame indices")
    parser.add_argument("--bbx_file", type=str, help="Path to store coordinates of bounding boxes")

    args = parser.parse_args()
    print(f"Received key_frames: {args.key_frames}")

    main(args.input, args.output, args.key_frames, args.bbx_file)