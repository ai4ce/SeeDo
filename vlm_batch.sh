#!/bin/bash

# 定义路径
CSV_FILE="/home/bw2716/VLMTutor/selected_valleys.csv"
VIDEO_DIR="/home/bw2716/VLMTutor/media/intermediate_demo"

# 循环读取 CSV 文件中的每一行
while IFS=, read -r demo indexlist
do
  # 去除 indexlist 中的多余字符（如双引号和回车符号）
  indexlist_python=$(echo "$indexlist" | sed 's/[\r""]//g')

  # 调试输出查看 indexlist_python 是否正确
  echo "Processed index list: $indexlist_python"

  # 构造视频路径
  video_path="$VIDEO_DIR/$demo.mp4"

  # 调用 Python 脚本处理每个视频
  python VLM_CaP/vlm.py --input "$video_path" --list "$indexlist_python" --demo "$demo"

done < <(tail -n +2 "$CSV_FILE")  # 跳过 CSV 文件的第一行（标题行）


