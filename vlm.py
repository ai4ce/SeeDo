import copy
import numpy as np
import cv2
import shapely
from shapely.geometry import *
from shapely.affinity import *
import matplotlib.pyplot as plt
from openai import OpenAI
from src.key import mykey, projectkey
import sys
from IPython.display import display, Image
import base64
from io import BytesIO
import os
import re
from PIL import Image
from collections import Counter
import argparse
import csv
import ffmpy
import ast
from src.vlm_video import extract_frame_list  # import extract_frames
# set up your openai api key
client = OpenAI(api_key=projectkey)
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
                # "Notice that there might be similar objects. You are supposed to use the index annotated on the objects to distinguish between only similar objects that is hard to distinguish with language."
def get_object_list(selected_frames):
    # first prompt to get objects in the environment
    prompt_messages_state = [
        {
            "role": "system",
            "content": [
                "You are a visual object detector. Your task is to count and identify the objects in the provided image that are on the desk. Focus on objects classified as grasped_objects and containers.",
            ],
        },
        {
            "role": "user",
            "content": [
                "There are two kinds of objects, grasped_objects and containers in the environment. We only care about objects on the desk. Do not count in hand or person as objects.",
                "Based on the input picture, answer:",
                "1. How many objects are there in the environment?",
                "2. What are these objects?",
                "You should respond in the format of the following example:",
                "Number: 1",
                "Objects: purple eggplant, red tomato, white bowl, white bowl",
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
def extract_keywords_reference(response):
    try:
        return response.split(': ')[1]
    except IndexError:
        print("Error extracting reference object from response:", response)
        return None
def is_frame_relevant(response):
    return "hand is manipulating an object" in response.lower()
def parse_closest_object_and_relationship(response):
    pattern = r"Closest Object: ([^,]+), (.+)"
    match = re.search(pattern, response)
    if match:
        return match.group(1), match.group(2)
    print("Error parsing reference object and relationship from response:", response)
    return None, None
def process_images(selected_frames, obj_list):
    string_cache = ""  # cache for CaP operations
    i = 1
    while i < len(selected_frames):
        input_frame_pick = selected_frames[i:i+1]
        prompt_messages_relevance_pick = [
            {
                "role": "system",
                "content": [
                    "You are an operations inspector. You need to check whether the hand in operation is holding an object. The objects have been outlined with contours of different colors and labeled with indexes for easier distinction."
                ],
            },
            {
                "role": "user",
                "content": [
                    "This is a picture from a pick-and-drop task. Please determine if the hand is manipulating an object.", 
                    "Respond with 'Hand is manipulating an object' or 'Hand is not manipulating an object'.",
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
                    "You are an operation inspector. You need to check which object is being picked in a pick-and-drop task. Some of the objects have been outlined with contours of different colors and labeled with indexes for easier distinction.",
                    "The contour and index is only used to help. Due to limitation of vision models, the contours and index labels might not cover every objects in the environment. If you notice any unannotated objects in the demo or in the object list, make sure you name it and handle them properly.",
                ],
            },
            {
                "role": "user",
                "content": [
                    f"This is a picture describing the pick state of a pick-and-drop task. The objects in the environment are {obj_list}. One of the objects is being picked by a human hand or robot gripper now. The objects have been outlined with contours of different colors and labeled with indexes for easier distinction.",
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
        # reference object 
        prompt_messages_reference = [
            {
                "role": "system",
                "content": [
                    "You are an operation inspector. You need to find the reference object for the placement location of the picked object in the pick-and-place process. Notice that the reference object can vary based on the task. If this is a storage task, the reference object should be the container into which the items are stored. If this is a stacking task, the reference object should be the object that best expresses the orientation of the arrangement."
                ],
            },
            {
                "role": "user",
                "content": [
                    f"This is a picture describing the drop state of a pick-and-place task. The objects in the environment are {obj_list}. {object_picked} is being dropped by a human hand or robot gripper now.",
                    "Based on the input picture and object list, answer:",
                    f"1. Which object in the rest of object list do you choose as a reference object to {object_picked}",
                    "You should respond in the format of the following example without any additional information or reason steps:",
                    "Reference Object: red block",
                    *map(lambda x: {"image": x, "resize": 768}, input_frame_drop),
                ],
            },
        ]
        response_reference = call_openai_api(prompt_messages_reference)
        print(response_reference)
        object_reference = extract_keywords_reference(response_reference)
        # current_bbx = bbx_list[i] if i < len(bbx_list) else {}
        
                    # "Due to limitation of vision models, the contours and index labels might not cover every objects in the environment. If you notice any unannotated objects in the demo or in the object list, make sure you handle them properly.",
        prompt_messages_relationship = [
            {
                "role": "system",
                "content": [
                    "You are a VLMTutor. You will describe the drop state of a pick-and-drop task from a demo picture. You must pay specific attention to the spatial relationship between picked object and reference object in the picture and be correct and accurate with directions.",
                ],
            },
            {
                "role": "user",
                "content": [
                    f"This is a picture describing the drop state of a pick-and-drop task. The objects in the environment are object list: {obj_list}. {object_picked} is said to be being dropped by a human hand or robot gripper now.",
                    f"However, the object being dropped might be wrong due to bad visual prompt. If you feel that object being picked is not {object_picked} but some other object, red chili is said to be the object picked but you feel it is an orange carrot, you MUST modify it and change the name!"
                    # "But notice that due to limitation of vision models, the contours and index labels might not cover every objects in the environment. If you notice any unannotated objects in the demo or in the object list, make sure you mention their name and handle their spatial relationships."
                    # "The ID is only used to help with your reasoning. You should only mention them when the objects are the same in language description. For example, when there are two white bowls, you must specify white bowl (ID:1), white bowl (ID:2) in your answer. But for different objects like vegetables, you do not need to specify their IDs."
                    # f"To help you better understand the spatial relationship, a bounding box list is given to you. Notice that the bounding boxes of objects in the bounding box list are distinguished by labels. These labels correspond one-to-one with the labels of the objects in the image. The bounding box list is: {bbx_list}",
                    # "The coordinates of the bounding box represent the center point of the object. The format is two coordinates (x,y). The origin of the coordinates is at the top-left corner of the image. If there are two objects A(x1, y1) and B(x2, y2), a significantly smaller x2 compared to x1 indicates that B is to the left of A; a significantly greater x2 compared to x1 indicates that B is to the right of A; a significantly smaller y2 compared to y1 indicates that B is at the back of A;  a significantly greater y2 compared to y1 indicates that B is in front of A."
                    # "Pay attention to distinguish between at the back of and on top of. If B and A has a visual gap, they are not in touch. Thus B is at the back of A. However, if they are very close, this means B and A are in contact, thus B is on top of A."
                    # "Notice that the largest difference in corresponding coordinates often represents the most significant feature. If you have coordinates with small difference in x but large difference in y, then coordinates y will represent most significant feature. Make sure to use the picture together with coordinates."
                    f"The object picked is being dropped somewhere near {object_reference}. Based on the input picture, object list answer:",
                    f"Drop object picked to which relative position to the {object_reference}? You need to mention the name of objects in your answer.",
                    f"There are totally six kinds of relative position, and the direction means the visual direction of the picture.",
                    f"1. In ((object picked is contained in the {object_reference})",
                    f"2. On top of (object picked is stacked on the {object_reference}, {object_reference} supports object picked)",
                    f"3. At the back of (in demo it means object picked is positioned farther to the viewer relative to the {object_reference})",
                    f"4. In front of (in demo it means object picked is positioned closer to the viewer or relative to the {object_reference})",
                    "5. to the left",
                    "6. to the right",
                    f"You must choose one relative position."
                    "You should respond in the format of the following example without any additional information or reason steps, be sure to mention the object picked and reference object.",
                    f"Drop yellow corn to the left of the red chili",
                    f"Drop red chili in the white bowl",
                    f"Drop wooden block (ID:1) to the right of the wooden block (ID:0)",
                    *map(lambda x: {"image": x, "resize": 768}, input_frame_drop),
                ],
            },
        ]
        response_relationship = call_openai_api(prompt_messages_relationship)
        print(response_relationship)
        string_cache += response_relationship + " and then "
        
        i += 1
        
    return string_cache
def save_results_to_csv(demo_name, num, obj_list, string_cache, output_file):
    """
    将结果保存到指定的 CSV 文件中，追加模式不会覆盖之前的内容。
    如果文件不存在，则创建并写入标题行。
    """
    file_exists = os.path.exists(output_file)
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # 如果文件不存在，则写入标题行
        if not file_exists:
            writer.writerow(["demo", "object", "action list"])
        
        # 写入数据
        writer.writerow([f"{demo_name}", f"{num} objects: {', '.join(obj_list)}", string_cache])
    print(f"Results appended to {output_file}")
def convert_video_to_mp4(input_path):
    """
    Converts the input video file to H.264 encoded .mp4 format using ffmpy.
    The output path will be the same as the input path with '_converted' appended before the extension.
    """
    # Get the file name without extension and append '_converted'
    base_name, ext = os.path.splitext(input_path)
    output_path = f"{base_name}_converted.mp4"
    # Run FFmpeg command to convert the video
    ff = ffmpy.FFmpeg(
        inputs={input_path: None},
        outputs={output_path: '-c:v libx264 -crf 23 -preset fast -r 30'}
    )
    ff.run()
    print(f"Video converted successfully: {output_path}")
    return output_path
def extract_bbx_list_from_csv(csv_file, input_video_path):
    """
    根据输入的视频路径，从CSV文件中提取对应的bbx列表，并按key_frame的index排序。
    """
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # 这里可以确保 input_video_path 与 row['demo'] 是否完全匹配
            if row['demo'] == input_video_path:
                # 提取bbx列，并按照key_frame的index排序
                bbx_info = row['bbx_list']
                # 提取出关键帧信息，格式为 key_frame<number>: <bounding_box_data>
                key_frame_data = re.findall(r'key_frame(\d+): (.+)', bbx_info)
                # 按关键帧 index 排序
                sorted_bbx_list = sorted(key_frame_data, key=lambda x: int(x[0]))
                # 处理排序后的信息并转换为 {index: bounding_box} 的形式
                bbx_list = []
                for _, objects in sorted_bbx_list:
                    # 匹配 Object 的 bounding box 数据，假设 x 和 y 是整数
                    object_bbx = re.findall(r'Object (\d+): \((\d+), (\d+)\)', objects)
                    # 如果需要支持浮点数，可以将 (\d+) 改为 ([\d.]+)
                    bbx_list.append({int(obj_index): (int(x), int(y)) for obj_index, x, y in object_bbx})
                return bbx_list  # 返回处理后的 bbx_list
    # 如果没有找到对应的 demo，返回 None
    return None
def main(input_video_path, frame_index_list, demo_name, csv_file):
    # 如果 frame_index_list 是字符串，使用 ast.literal_eval 将其转换为列表
    # 现在转换为整数列表
    frame_index_list = ast.literal_eval(frame_index_list)
    # 从csv_file中提取与input_video_path对应的bbx_list
    # bbx_list = extract_bbx_list_from_csv(csv_file, input_video_path)
    # if not bbx_list:
    #     print(f"No matching bbx list found for {input_video_path}")
    #     return
    # video path
    video_path = input_video_path
    # list to store key frames
    selected_raw_frames1 = []
    # list to store key frame indexes
    selected_frame_index = frame_index_list
    # Convert the video to H.264 encoded .mp4 format
    # converted_video_path = convert_video_to_mp4(video_path)
    # Open the converted video
    cap = cv2.VideoCapture(video_path)
    # Manually calculate total number of frames
    actual_frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        actual_frame_count += 1
    # Reset the capture to the beginning of the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"Actual frame count: {actual_frame_count}")
    # Iterate through index list and get the frame list
    for index in selected_frame_index:
        if index < actual_frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, cv2_image = cap.read()
            if ret:
                selected_raw_frames1.append(cv2_image)
            else:
                print(f"Failed to retrieve frame at index {index}")
        else:
            print(f"Frame index {index} is out of range for this video.")
    # Release video capture object
    cap.release()
    # 调用处理函数
    selected_frames1 = extract_frame_list(selected_raw_frames1)
    response_state = get_object_list(selected_frames1)
    num, obj_list = extract_num_object(response_state)
    print("Number of objects:", num)
    print("available objects:", obj_list)
    # obj_list = "green corn, orange carrot, red pepper, white bowl, glass container"
    # process the key frames
    string_cache = process_images(selected_frames1, obj_list)
    if string_cache.endswith(" and then "):
        my_string = string_cache.removesuffix(" and then ")
    results_file = "./new_vegetable_results.csv"
    save_results_to_csv(demo_name, num, obj_list, string_cache, results_file)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video and key frame extraction.")
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--list', type=str, required=True, help='List of key frame indexes')
    parser.add_argument('--bbx_csv', type=str, required=True, help='csv of bbx')
    parser.add_argument('--demo', type=str, required=True, help='demo name')
    args = parser.parse_args()
    # Call the main function with arguments
    main(args.input, args.list, args.demo, args.bbx_csv)