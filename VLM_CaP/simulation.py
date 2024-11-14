import argparse
import copy
import numpy as np
from src.env import PickPlaceEnv
from src.LMP import LMP, LMP_wrapper, LMPFGen
from src.configs import cfg_tabletop, lmp_tabletop_coords
from src.key import projectkey
from openai import OpenAI
import shapely
from shapely.geometry import *
from shapely.affinity import *
from moviepy.editor import ImageSequenceClip, concatenate_videoclips

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

def execute_actions(action_list, obj_list, env, lmp_tabletop_ui, output_path):
    # Split action_list into individual tasks
    tasks = action_list.split("and then")
    
    # List to hold all video clips
    video_clips = []

    # Process each task separately
    for task in tasks:
        env.cache_video = []  # Clear the cache for the new task
        print(f"Running task: {task.strip()} and recording video...")
        lmp_tabletop_ui(task.strip(), f'objects = {env.object_list}')
        
        # Render the video for the task
        if env.cache_video:
            task_clip = ImageSequenceClip(env.cache_video, fps=30)
            video_clips.append(task_clip)

    # Concatenate all the task videos into one final video
    if video_clips:
        final_clip = concatenate_videoclips(video_clips, method="compose")
        final_clip.write_videofile(output_path, codec='libx264', bitrate="5000k", fps=30)
        print(f"Final video saved at {output_path}")

def main(args):
    client = OpenAI(api_key=projectkey)
    # Initialize environment and LMP with passed arguments
    obj_list = args.obj_list
    action_list = args.action_list
    output_path = args.output  # Output path for final video

    # Initialize environment
    env = PickPlaceEnv(render=True, high_res=True, high_frame_rate=False)
    _ = env.reset(obj_list)
    lmp_tabletop_ui = setup_LMP(env, cfg_tabletop, client)

    # Execute actions and save video
    execute_actions(action_list, obj_list, env, lmp_tabletop_ui, output_path)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run PickPlaceEnv with LMP based on action list.")
    parser.add_argument('--action_list', type=str, required=True, help='String of actions separated by "and then"')
    parser.add_argument('--obj_list', nargs='+', required=True, help='List of object names in the environment')
    parser.add_argument('--output', type=str, required=True, help='Path to save the final video')

    args = parser.parse_args()

    main(args)