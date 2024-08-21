# video_cropper.py
import cv2

def crop_video(video_path, x, y, w, h):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 绘制矩形框以显示裁剪区域
        frame_with_rect = frame.copy()
        cv2.rectangle(frame_with_rect, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 显示带有裁剪区域的帧
        cv2.imshow('Frame with proposed ROI', frame_with_rect)
        key = cv2.waitKey(1)
        if key == ord('q'):  # 按 'q' 键退出
            break
        elif key == ord('c'):  # 按 'c' 键裁剪并显示裁剪区域
            roi = frame[y:y+h, x:x+w]
            cv2.imshow('Cropped ROI', roi)

    cap.release()
    cv2.destroyAllWindows()

def crop_video_to_video(video_path, x, y, w, h, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取原视频的帧率
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码器
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))  # 创建VideoWriter对象

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 裁剪帧的感兴趣区域
        cropped_roi = frame[y:y+h, x:x+w]
        out.write(cropped_roi)  # 写入帧

    cap.release()
    out.release()  # 释放VideoWriter对象