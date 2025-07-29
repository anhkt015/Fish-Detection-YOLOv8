
#code dự đoán dựa trên video 
!pip install ultralytics opencv-python
!pip install ultralytics --upgrade
import yt_dlp
import cv2
from ultralytics import YOLO

# Link video YouTube bạn muốn dùng
video_url = 'https://youtu.be/7qRpcTUwYHM?si=BzJC6igu0rZLszNN'  # Thay bằng link bạn muốn
video_filename = 'cavang_video.mp4'
output_avi_filename = 'output.avi'

# Tải video bằng yt-dlp
ydl_opts = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    'outtmpl': video_filename,
    'merge_output_format': 'mp4',
}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])

model = YOLO('/content/drive/MyDrive/best.pt')
cap = cv2.VideoCapture(video_filename)

# Use a common codec for AVI (e.g., DIVX or XVID) - writing to .avi first
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_avi_filename, fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, conf=0.25)
    annotated = results[0].plot()
    out.write(annotated)

cap.release()
out.release()