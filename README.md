# SeeDo: Human Demo Video to Robot Action Plan via Vision Language Model

**VLM See, Robot Do (SeeDo)** is a method that uses large vision models, tracking models and vision-language models to extract robot action plan from human demonstration video, specifically focusing on long horizon pick-and-place tasks. The action plan is then implemented in realworld and PyBullet simulation environment.

![main](https://github.com/ai4ce/SeeDo/blob/main/media/main.jpg)

## Setup Instructions

Note that SeeDo relies on GroundingDINO and SAM2.

- Install SeeDo and create a new environment

```python
git clone https://github.com/ai4ce/SeeDo
conda create --name seedo python=3.10
conda activate seedo
pip install -r requirements.txt
```

- Install GroundingDINO and SAM2 in the same environment

```python
git clone https://github.com/IDEA-Research/GroundingDINO
git clone https://github.com/facebookresearch/sam2
```

- Obtain an OpenAI API key and create a `key.py` file under `VLM_CaP/src`

```python
cd VLM_CaP/src
touch key.py
echo 'projectkey = "YOUR_OPENAI_API_KEY"' > key.py
```

## Pipeline

There are mainly four parts of SeeDo. To ensure the video is successfully processed in subsequent steps, use `convert_video.py` to convert the video to the appropriate encoding before inputting it. The `convert_video.py` script accepts two parameters: `--input` and `--output`, which specify the path of your original video and the path of the converted video, respectively.
*Currently, this repo only supports the code of the first three modules*

1. **Keyframe Selection Module**

   `get_frame_by_hands.py`: The `get_frame_by_hands.py` script allows selecting key frames by tracking hand movements. It accepts two parameters.

    `--video_path`, which specifies the path of the input video.

   `--output_dir`, which designates the directory where the key frames will be saved. If `output_dir` is not specified, the keyframes will be saved to `./output` by default. For debugging purpose, the hand image and hand speed plot will also be saved in this directory.

2. **Visual Perception Module**

   `track_objects.py`: The `track_objects.py` script is used to track each object and add a visual prompt for the objects. It also returns a string containing the center coordinates of each object in the key frames. The script accepts three parameters.

    `--input` is the video converted to the appropriate format.

    `--output` specifies the output path for the video with the visual prompts. 

    `--key_frames` is the list of key frame indices obtained from `get_frames_by_hands.py`.

   This module will return a `box_list` string stored for useage in  VLM Reasoning Module

3. **VLM Reasoning Module**

   `vlm.py`: The `vlm.py` script performs reasoning on the key frames and generates an action list for the video. It accepts three parameters.

    `--input` is the video with visual prompts added by the Visual Perception Module.

    `--list` is the keyframe index list obtained from the Keyframe Selection Module.

   `--bbx_list` is the `box_list` string obtained from the Visual Perception Module.

   This module will return two strings: `obj_list` representing for the objects in the environment; `action_list` representing for the action list performed on these objects.

4. **Robot Manipulation Module**

   `simulation.py`: The `simulation.py` script accepts two parameters: `obj_list` and `action_list`. It first initializes a random simulation scene based on the `obj_list`, and then executes pick-and-place tasks according to the `action_list`. 

   Note that this part uses a modified version of the Code as Policies framework, and its successful execution depends heavily on whether the objects are already modeled and whether the corresponding execution functions for actions are present in the prompt. We provide a series of new object models and prompts that are compatible with our defined action list. If you want to operate on unseen objects, you will need to provide the corresponding object modeling.

   It will write out a video of robot manipulation of a series of pick-and-place tasks in simulation.
