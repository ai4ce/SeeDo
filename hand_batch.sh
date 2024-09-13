#!/bin/bash

# Set the base paths
VIDEO_PATH_BASE="/home/bw2716/VLMTutor/media/input_demo/new_wooden_block/wooden_block"
OUTPUT_DIR="/home/bw2716/VLMTutor/media/intermediate_demo/new_wooden_block"
# Loop from long_demo1 to long_demo30
for i in {29..39}
do
    # Construct the video file path with zero-padding for single-digit numbers
    VIDEO_PATH="${VIDEO_PATH_BASE}${i}.mp4"
    
    # Run the Python script with the current video path and constant output directory
    echo "Processing video: ${VIDEO_PATH}"
    python get_frame_by_hands.py --video_path "$VIDEO_PATH" --output_dir "$OUTPUT_DIR"
done

echo "All videos processed."