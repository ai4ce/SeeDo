# 定义输入和输出文件夹
input_folder="/home/bw2716/VLMTutor/media/input_demo/new_wooden_block/"
output_folder="/home/bw2716/VLMTutor/media/output_demo/new_wooden_block/"
csv_file="/home/bw2716/VLMTutor/new_wooden_block_selected_valleys.csv"

# 遍历 long_demo1.mp4 到 long_demo20.mp4
for i in {1..29}; do
  input_video="${input_folder}wooden_block${i}.mp4"
  output_video="${output_folder}wooden_block${i}_contour_num.mp4"

  # 获取当前 demo 名字对应的 key_frames 列表
  demo_name="wooden_block${i}_contour_num.mp4"
  
  # 调试输出，检查是否正确匹配到 demo 名字
  echo "Searching for $demo_name in CSV file..."
  grep_output=$(grep "$demo_name" "$csv_file")
  
  # 调试输出：显示 grep 提取的内容
  echo "grep result: $grep_output"

  # 提取 key_frames，使用 sed 删除逗号前的部分
  key_frames=$(echo "$grep_output" | sed 's/^[^,]*,//')
  
  # 调试输出：检查提取的 key_frames 原始值
  echo "Raw key_frames value: $key_frames"

  # 移除引号和方括号并转换为一个空格分隔的列表
  key_frames=$(echo $key_frames | tr -d '[]\"' | tr ',' ' ' | xargs)

  # 在这里添加调试输出以查看提取的 key_frames 列表是否正确
  echo "Extracted key_frames for $demo_name: $key_frames"

  if [ -f "$input_video" ]; then
    echo "Processing $input_video with key_frames $key_frames..."
    # 直接传递 key_frames 作为多个参数
    python track_objects.py --input "$input_video" --output "$output_video" --key_frames $key_frames
  else
    echo "File $input_video does not exist."
  fi
done
