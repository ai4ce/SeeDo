#!/bin/bash

# Set the base paths
VIDEO_PATH_BASE="/home/bw2716/VLMTutor/media/intermediate_demo/short_demo"
OUTPUT_DIR="/home/bw2716/VLMTutor/media/output_demo/fruit_container_task_sam2contour"
# Loop from long_demo1 to long_demo30
for i in {1..30}
do
    # Construct the video file path with zero-padding for single-digit numbers
    VIDEO_PATH="${VIDEO_PATH_BASE}${i}_sam2_contour.mp4"
    
    # Run the Python script with the current video path and constant output directory
    echo "Processing video: ${VIDEO_PATH}"
    python get_frame_by_hands.py --video_path "$VIDEO_PATH" --output_dir "$OUTPUT_DIR"
done

echo "All videos processed."