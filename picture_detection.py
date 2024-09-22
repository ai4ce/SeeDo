import torch
from tqdm import *
import cv2
from openai import OpenAI
import base64
from collections import Counter
import torch
from huggingface_hub import hf_hub_download

# Grounding DINO
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict, load_image_from_array

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model

# Grounding DINO
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict, load_image_from_array

def bbx_extract(groundingdino_model, DEVICE, input_image, obj_list):
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    BOX_THRESHOLD = 0.3
    TEXT_THRESHOLD = 0.25
    object_counts = Counter(obj_list)
    
    image_source, image = load_image_from_array(input_image)

    best_boxes = []
    best_phrases = []
    best_logits = []

    # Iterate through each object and select the bounding box with the highest confidence
    for obj, count in object_counts.items():
        boxes, logits, phrases = predict(
            model=groundingdino_model,
            image=image,
            caption=obj,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE
        )

        if boxes.shape[0] > 0:
            # Select bounding boxes based on the number of times the object appears in obj_list
            selected_count = min(count, boxes.shape[0])
            for i in range(selected_count):
                best_boxes.append(boxes[i].unsqueeze(0))
                best_phrases.append(phrases[i])
                best_logits.append(logits[i])
    return image_source,best_boxes,best_phrases,best_logits

def call_openai_api(prompt_messages, client):
    params = {
        "model": "gpt-4o",
        "messages": prompt_messages,
        "max_tokens": 400,
        "temperature": 0
    }
    result = client.chat.completions.create(**params)
    return result.choices[0].message.content


def get_object_list(video_path, client):
    # Encode the first frame
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    frame_count = 0
    max_frames = 2  # Process the first two frames

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
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0:1]),  # Use the first picture for environment objects
            ],
        },
    ]

    response_state = call_openai_api(prompt_messages_state, client)
    return response_state