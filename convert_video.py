import ffmpy
import argparse

def convert_video_to_30fps(input_path, output_path):
    """
    Converts the input video file to H.264 encoded .mp4 format with 30 FPS using ffmpy.
    """
    ff = ffmpy.FFmpeg(
        inputs={input_path: None},
        outputs={output_path: '-c:v libx264 -r 30 -crf 23 -preset fast'}
    )
    ff.run()
    print(f"Video converted successfully to 30 FPS: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert a video to 30 FPS with H.264 encoding.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--output', type=str, required=True, help="Path to the output video file.")
    
    args = parser.parse_args()

    # Convert the single video
    convert_video_to_30fps(args.input, args.output)

if __name__ == '__main__':
    main()