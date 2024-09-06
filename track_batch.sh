#!/bin/bash

# 定义输入和输出文件夹
input_folder="/home/bw2716/VLMTutor/media/input_demo/fruit_container_demo/"
output_folder="/home/bw2716/VLMTutor/media/intermediate_demo/"

# 遍历 long_demo1.mp4 到 long_demo30.mp4
for i in {1..30}; do
  input_video="${input_folder}long_demo${i}.mp4"
  output_video="${output_folder}long_demo${i}_sam2_contour.mp4"

  if [ -f "$input_video" ]; then
    echo "Processing $input_video..."
    # 调用你的 Python 脚本并传递 input_video 和 output_video 参数
    python track_objects.py --input "$input_video" --output "$output_video"
    echo "Finished processing $output_video"
  else
    echo "File $input_video does not exist."
  fi
done
