import ffmpy
import os

def convert_video_to_30fps(input_path, output_path):
    """
    Converts the input video file to H.264 encoded .mp4 format with 30 FPS using ffmpy.
    """
    ff = ffmpy.FFmpeg(
        inputs={input_path: None},
        outputs={output_path: '-c:v libx264 -r 30 -crf 23 -preset fast'}
    )
    ff.run()
    print(f"Video converted successfully to 15 FPS: {output_path}")

def batch_convert_videos_to_30fps(input_dir, output_dir, start_block, end_block):
    """
    Batch converts videos from block1 to blockN and sets frame rate to 30 FPS.
    """
    for i in range(start_block, end_block + 1):
        input_file = os.path.join(input_dir, f'garments_organization{i}.mp4')
        output_file = os.path.join(output_dir, f'garments_organization{i}_30fps.mp4')

        if os.path.exists(input_file):
            convert_video_to_30fps(input_file, output_file)
        else:
            print(f"Input file does not exist: {input_file}")

# 设置输入输出目录，以及起始和结束block编号
input_directory = "/home/bw2716/VLMTutor/media/input_demo/garments_organization/"
output_directory = "/home/bw2716/VLMTutor/media/input_demo/garments_organization/"

# 如果输出目录不存在，则创建
os.makedirs(output_directory, exist_ok=True)

# 批量转换 block1 到 block14 的视频并设置帧率为30FPS
batch_convert_videos_to_30fps(input_directory, output_directory, 3, 3)