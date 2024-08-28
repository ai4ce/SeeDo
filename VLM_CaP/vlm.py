import copy
import numpy as np
from src.env import PickPlaceEnv
from src.env import ALL_BLOCKS, ALL_BOWLS
from src.LMP import LMP, LMP_wrapper, LMPFGen
from src.configs import cfg_tabletop, lmp_tabletop_coords
import cv2
import shapely
from shapely.geometry import *
from shapely.affinity import *
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import moviepy
from openai import OpenAI
from src.key import mykey
import sys
from IPython.display import display, Image
import base64
from io import BytesIO
import os
import re
from PIL import Image

from src.vlm_video import extract_frame_list  # import extract_frames

def setup_LMP(env, cfg_tabletop, openai_client):
    # LMP env wrapper
    cfg_tabletop = copy.deepcopy(cfg_tabletop)
    cfg_tabletop["env"] = dict()
    cfg_tabletop["env"]["init_objs"] = list(env.obj_name_to_id.keys())
    cfg_tabletop["env"]["coords"] = lmp_tabletop_coords
    LMP_env = LMP_wrapper(env, cfg_tabletop)
    # creating APIs that the LMPs can interact with
    fixed_vars = {"np": np}
    fixed_vars.update(
        {
            name: eval(name)
            for name in shapely.geometry.__all__ + shapely.affinity.__all__
        }
    )
    variable_vars = {
        k: getattr(LMP_env, k)
        for k in [
            "get_bbox",
            "get_obj_pos",
            "get_color",
            "is_obj_visible",
            "denormalize_xy",
            "put_first_on_second",
            "get_obj_names",
            "get_corner_name",
            "get_side_name",
        ]
    }
    variable_vars["say"] = lambda msg: print(f"robot says: {msg}")

    # creating the function-generating LMP
    lmp_fgen = LMPFGen(openai_client, cfg_tabletop["lmps"]["fgen"], fixed_vars, variable_vars)

    # creating other low-level LMPs
    variable_vars.update(
        {
            k: LMP(openai_client, k, cfg_tabletop["lmps"][k], lmp_fgen, fixed_vars, variable_vars)
            for k in [
                "parse_obj_name",
                "parse_position",
                "parse_question",
                "transform_shape_pts",
            ]
        }
    )

    # creating the LMP that deals w/ high-level language commands
    lmp_tabletop_ui = LMP(
        openai_client,
        "tabletop_ui",
        cfg_tabletop["lmps"]["tabletop_ui"],
        lmp_fgen,
        fixed_vars,
        variable_vars,
    )

    return lmp_tabletop_ui

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
    prompt_messages_state = [
        {
            "role": "system",
            "content": [
                "You are a visual object detector. Your task is to count and identify the objects in the provided image that are on the desk. Focus on objects classified as grasped_objects and containers."
            ],
        },
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
                *map(lambda x: {"image": x, "resize": 768}, selected_frames[0:1]),  # use first picture for environment objects
            ],
        },
    ]
    response_state = call_openai_api(prompt_messages_state)
    return response_state

def extract_num_object(response_state):
    # extract number of objects
    num_match = re.search(r"Number: (\d+)", response_state)
    num = int(num_match.group(1)) if num_match else 0
    
    # extract objects
    objects_match = re.search(r"Objects: (.+)", response_state)
    objects_list = objects_match.group(1).split(", ") if objects_match else []
    
    # construct object list
    objects = [obj for obj in objects_list]
    
    return num, objects

def extract_keywords_pick(response):
    try:
        return response.split(': ')[1]
    except IndexError:
        print("Error extracting pick keyword from response:", response)
        return None

def extract_keywords_drop(response):
    try:
        return response.split(': ')[1]
    except IndexError:
        print("Error extracting drop keyword from response:", response)
        return None

def extract_keywords_closest(response):
    try:
        return response.split(': ')[1]
    except IndexError:
        print("Error extracting closest object from response:", response)
        return None

def is_frame_relevant(response):
    return "hand is holding an object" in response.lower()

def parse_closest_object_and_relationship(response):
    pattern = r"Closest Object: ([^,]+), (.+)"
    match = re.search(pattern, response)
    if match:
        return match.group(1), match.group(2)
    print("Error parsing closest object and relationship from response:", response)
    return None, None

def process_images(selected_frames):
    string_cache = ""  # cache for CaP operations
    i = 1

    while i < len(selected_frames):
        # Check if the first frame (i) is relevant (i.e., hand is holding an object)
        input_frame_pick = selected_frames[i:i+1]
        prompt_messages_relevance_pick = [
            {
                "role": "system",
                "content": [
                    "You are an operations inspector. You need to check whether the hand in operation is holding an object."
                ],
            },
            {
                "role": "user",
                "content": [
                    "This is a picture from a pick-and-drop task. Please determine if the hand is holding an object.", 
                    "Respond with 'Hand is holding an object' or 'Hand is not holding an object'.",
                    *map(lambda x: {"image": x, "resize": 768}, input_frame_pick),
                ],
            },
        ]
        response_relevance_pick = call_openai_api(prompt_messages_relevance_pick)
        print(response_relevance_pick)
        if not is_frame_relevant(response_relevance_pick):
            i += 1
            continue

        # which to pick
        prompt_messages_pick = [
            {
                "role": "system",
                "content": [
                    "You are an operation inspector. You need to check which object is being picked in a pick-and-drop task."
                ],
            },
            {
                "role": "user",
                "content": [
                    f"This is a picture describing the pick state of a pick-and-drop task. The objects in the environment are {obj_list}. One of the objects is being picked by a human hand or robot gripper now.",
                    "Based on the input picture and object list, answer:",
                    "1. Which object is being picked",
                    "You should respond in the format of the following example:",
                    "Object Picked: red block",
                    *map(lambda x: {"image": x, "resize": 768}, input_frame_pick),
                ],
            },
        ]
        response_pick = call_openai_api(prompt_messages_pick)
        print(response_pick)
        object_picked = extract_keywords_pick(response_pick)

        i += 1

        # Ensure there is another frame for drop and relative position reasoning
        if i >= len(selected_frames):
            break

        # Check if the second frame (i) is relevant (i.e., hand is holding an object)
        input_frame_drop = selected_frames[i:i+1]
        prompt_messages_relevance_drop = [
            {
                "role": "system",
                "content": [
                    "You are an operations inspector. You need to check whether the hand in operation is holding an object."
                ],
            },
            {
                "role": "user",
                "content": [
                    "This is a picture from a pick-and-drop task. Please determine if the hand is holding an object.",
                    "Respond with 'Hand is holding an object' or 'Hand is not holding an object'.",
                    *map(lambda x: {"image": x, "resize": 768}, input_frame_drop),
                ],
            },
        ]
        response_relevance_drop = call_openai_api(prompt_messages_relevance_drop)
        print(response_relevance_drop)
        if not is_frame_relevant(response_relevance_drop):
            i += 1
            continue

        # closest object 
        prompt_messages_closest = [
            {
                "role": "system",
                "content": [
                    "You are an operation inspector. You need to find the reference object for the placement location of the picked object in the pick-and-place process."
                ],
            },
            {
                "role": "user",
                "content": [
                    f"This is a picture describing the drop state of a pick-and-place task. The objects in the environment are {obj_list}. {object_picked} is being dropped by a human hand or robot gripper now.",
                    "Based on the input picture and object list, answer:",
                    f"1. Which object in the rest of object list is closest to {object_picked}",
                    "You should respond in the format of the following example without any additional information or reason steps:",
                    "Closest Object: red block",
                    *map(lambda x: {"image": x, "resize": 768}, input_frame_drop),
                ],
            },
        ]
        response_closest = call_openai_api(prompt_messages_closest)
        print(response_closest)
        object_closest = extract_keywords_closest(response_closest)

        # and relative position
        prompt_messages_relationship = [
            {
                "role": "system",
                "content": [
                    "You are a VLMTutor. You will describe the drop state of a pick-and-drop task from a demo picture. You must pay specific attention to the spatial relationship between interested objects in the picture and be correct and accurate with directions.",
                    "Be careful with the word adjacent. You must only use this word when the objects are physically in touch.",
                ],
                "role": "user",
                "content": [
                    f"This is a picture describing the drop state of a pick-and-drop task. The objects in the environment are object list: {obj_list}. {object_picked} is being dropped by a human hand or robot gripper now.",
                    f"It is being dropped somewhere near {object_closest}. Based on the input picture and object list, answer:",
                    f"Drop {object_picked} to which relative position from the {object_closest}? You need to mention the name of objects in your answer",
                    "There are totally ten kinds of relative position, and the direction means the visual direction of the picture",
                    f"1. In (({object_picked} is contained in the {object_closest})"
                    f"2. above ({object_picked} is located in higher position than the {object_closest}, and are not in contact or overlaoped)",
                    f"3. adjacent above ({object_picked} is located in higher position than the {object_closest}, and are in contact or overlapped)",
                    f"4. below ({object_picked} is located in lower position than the {object_closest}, and are not in contact or overlapped)",
                    f"5. adjacent below ({object_picked} is located in lower position than the {object_closest}, and are in contact or overlapped)",
                    f"6. On top of ({object_picked} is stacked on the {object_closest}, {object_closest} supports {object_picked})",
                    "7. to the left (meaning the two blocks are not in contact or overlapped)",
                    "8. to the right (meaning the two blocks are not in contact or overlapped)",
                    "9. adjacent to the left (meaning the two blocks are in contact or overlapped)",
                    "10. adjacent to the right (meaning the two blocks are in contact or overlapped)",
                    "You should respond in the format of the following example without any additional information or reason steps, be sure to mention the object picked and closest object",
                    f"Drop {object_picked} to the left of the {object_closest}",
                    *map(lambda x: {"image": x, "resize": 768}, input_frame_drop),
                ],
            },
        ]
        response_relationship = call_openai_api(prompt_messages_relationship)
        print(response_relationship)
        string_cache += response_relationship + " and then "
        
        i += 1
        
    return string_cache

# set up your openai api key
client = OpenAI(api_key=mykey)

###################################################################################
# 选择帧路径
folder_path = '/home/bw2716/VLMTutor/media/output_demo/fruit_container_task/long_demo1/selected_frames'
# 存储图像帧的列表
selected_raw_frames1 = []
# 获取并排序文件夹中的所有图像
filenames = sorted(
    os.listdir(folder_path),
    key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0])))
)
# 遍历排序后的文件名列表
for filename in filenames:
    if filename.endswith('.jpg') or filename.endswith('.png'):
        file_path = os.path.join(folder_path, filename)
        # 使用 OpenCV 读取图像
        cv2_image = cv2.imread(file_path)
        selected_raw_frames1.append(cv2_image)

selected_frames1 = extract_frame_list(selected_raw_frames1)

# 显示前5帧
# for i in range(min(5, len(selected_frames1))):
#     plt.figure()
#     plt.imshow(cv2.cvtColor(selected_frames1[i], cv2.COLOR_BGR2RGB))
#     plt.title(f"Frame {i}")
#     plt.axis('off')  # 隐藏坐标轴
#     plt.show()

#########################################################################
# first get the object list
response_state = get_object_list(selected_frames1)
num, obj_list = extract_num_object(response_state)
print("Number of objects:", num)
print("available objects:", obj_list)
# obj_list = "green corn, orange carrot, red pepper, white bowl, glass container"
# process the key frames
string_cache = process_images(selected_frames1)
if string_cache.endswith(" and then "):
    my_string = string_cache.removesuffix(" and then ")

# check the action list
print("action list: " + string_cache)
print("Current working directory:", os.getcwd())
# replace urdf_path with your local path
urdf_path = "VLM_CaP/ur5e/ur5e.urdf"
# initialize environment
env = PickPlaceEnv(render=True, high_res=True, high_frame_rate=False)
_ = env.reset(obj_list)
lmp_tabletop_ui = setup_LMP(env, cfg_tabletop, client)

# check again for the objects
print('available objects:')
print(obj_list)

user_input = string_cache
env.cache_video = []

print('Running policy and recording video...')
lmp_tabletop_ui(user_input, f'objects = {env.object_list}')

# render video
if env.cache_video:
    rendered_clip = ImageSequenceClip(env.cache_video, fps=25)
    video_path = './rendered_video.mp4'
    rendered_clip.write_videofile(video_path, codec='libx264')