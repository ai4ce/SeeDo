# SeeDo: Human Demo Video to Robot Action Plan via Vision Language Model
<a href='https://arxiv.org/abs/2410.08792'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a> <a href='https://ai4ce.github.io/SeeDo/'><img src='https://img.shields.io/badge/Project-website-green'></a>

**VLM See, Robot Do (SeeDo)** is a method that uses large vision models, tracking models and vision-language models to extract robot action plans from human demonstration videos, specifically focusing on long horizon pick-and-place tasks. The action plan is then implemented in real-world and PyBullet simulation environments.

![main](https://github.com/ai4ce/SeeDo/blob/main/media/main.jpg)

## News
- [2025/06] SeeDo is accepted by IROS 2025! We will update the camera-ready version soon.

## Setup Instructions

Note that SeeDo relies on GroundingDINO, SAM and SAM2. The code has only been tested on Ubuntu 20.04. The version of CUDA tested is 11.8, the Pytorch version is 2.3.1+cu118.

- Install SeeDo and create a new environment

```python
git clone https://github.com/ai4ce/SeeDo
conda create --name seedo python=3.10.14
conda activate seedo
cd SeeDo
pip install -r requirements.txt
```

- Install Pytorch (Only for CUDA 11.8 user)

```python
pip install torch==2.3.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- Install GroundingDINO, SAM and SAM2 in the same environment

```python
git clone https://github.com/IDEA-Research/GroundingDINO
git clone https://github.com/facebookresearch/segment-anything.git
git clone https://github.com/facebookresearch/segment-anything-2.git
```

- Make sure these models are installed in editable packages

```python
cd GroundingDINO
pip install -e .
```
And do the same with segment-anything, segment-anything-2

- We have slightly modified the GroundingDINO

In `GroundingDINO/groundingdino/util/inference.py`, we add a function to help inference on an array of images. Please paste the following function into `inference.py`.

```python
def load_image_from_array(image_array: np.array) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.fromarray(image_array)
    image_transformed, _ = transform(image_source, None)
    return image_array, image_transformed
```

- The code still uses one checkpoint from segment-anything.

Make sure you download it in the SeeDo folder.
**`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**

- Obtain an OpenAI API key and create a `key.py` file under `VLM_CaP/src`

```python
cd VLM_CaP/src
touch key.py
echo 'projectkey = "YOUR_OPENAI_API_KEY"' > key.py
```

## Pipeline

There are mainly four parts of SeeDo. To ensure the video is successfully processed in subsequent steps, use `convert_video.py` to convert the video to the appropriate encoding before inputting it. The `convert_video.py` script accepts two parameters: `--input` and `--output`, which specify the path of your original video and the path of the converted video, respectively.

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

   `simulation.py`: The `simulation.py` script accepts three parameters: `obj_list`, `action_list`, `output`. It first initializes a random simulation scene based on the `obj_list`, and then executes pick-and-place tasks according to the `action_list`, and finally write the video to output.

   Example usage: `python simulation.py --action_list "put chili on bowl and then put eggplant on glass" --obj_list chili carrot eggplant bowl glass --output demo2.mp4`

   Note that this part uses a modified version of the Code as Policies framework, and its successful execution depends heavily on whether the objects are already modeled and whether the corresponding execution functions for actions are present in the prompt. We provide a series of new object models and prompts that are compatible with our defined action list. If you want to operate on unseen objects, you will need to provide the corresponding object modeling, and modify the LMP and prompt file accordingly.

   We provide some simple object modelings of vegetables on hugging face. Download from https://huggingface.co/datasets/ai4ce/SeeDo/tree/main/SeeDo

   There will be an `assets.zip` file, extract that file into `assets` and make sure this folder is under the path of VLM_CaP. `VLM_CaP/assets` will then be used by `simulation.py` for simulation.

   It will write out a video of robot manipulation of a series of pick-and-place tasks in simulation.
