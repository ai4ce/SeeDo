import subprocess
import os

def run_track_objects_script():
    # 运行 test.py 并获取输出视频路径
    process = subprocess.Popen(['python', 'track_objects.py'], stdout=None, stderr=None)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error running track_objects.py: {stderr.decode()}")
        return None
    return '/home/bw2716/VLMTutor/Cropped_real_world_demo_track.mp4'

def run_get_frame_by_hands_script(video_path):
    # 运行 get_frame_by_hands.py 并传递视频路径作为参数
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    process = subprocess.Popen(['python', 'get_frame_by_hands.py', '--video_path', video_path, '--output_dir', './'], stdout=None, stderr=None)
    stdout, stderr = process.communicate()
    # if process.returncode != 0:
    #     print(f"Error running get_frame_by_hands.py: {stderr.decode()}")
    # else:
    #     print(stdout.decode())

def run_vlm_script():
    # 运行 vlm.py
    process = subprocess.Popen(['python', 'VLM_CaP/vlm.py'], stdout=None, stderr=None)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error running vlm.py: {stderr.decode()}")
    else:
        print(stdout.decode())

def main():
    output_video_path = run_track_objects_script()
    if output_video_path:
        run_get_frame_by_hands_script(output_video_path)
        run_vlm_script()

if __name__ == "__main__":
    main()

