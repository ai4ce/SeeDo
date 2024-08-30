import os
import subprocess

# Define the input and output directories
input_dir = '/home/bw2716/VLMTutor/media/input_demo/fruit_container_demo/'
output_dir = '/home/bw2716/VLMTutor/media/intermediate_demo/'

# Get a list of all .mp4 files in the input directory
mp4_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]

# Loop through each .mp4 file
for mp4_file in mp4_files:
    # Construct the full input file path
    input_file_path = os.path.join(input_dir, mp4_file)
    
    # Construct the output file path by appending '_sam2_contour' to the filename
    output_file_path = os.path.join(output_dir, mp4_file.replace('.mp4', '_sam2_contour.mp4'))
    
    # Run the track_objects.py script with the input and output paths
    subprocess.run(['python3', 'track_objects.py', '--video_path', input_file_path, '--output_video_path', output_file_path])

print("Processing complete.")
