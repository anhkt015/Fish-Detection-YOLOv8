# Fish-Detection-YOLOv8
A YOLOv8-based fish detection project using custom dataset on colab .

## 🧠 Mô hình
- Mô hình: YOLOv8n
- Số epoch huấn luyện: 25
-số lượng ảnh : 4100
- Nguồn dữ liệu: Roboflow (
from roboflow import Roboflow
rf = Roboflow(api_key="ox2ikPVhbk16eKvjBy97")
project = rf.workspace("hackertrip").project("detection-of-fish-ehwb6")
version = project.version(1)
dataset = version.download("yolov8")
)

## 📁 File trong gg drive
-https://drive.google.com/drive/folders/1o7JvFhhiZMdNRNXmQUx-J5jo2oiiGEjO?usp=sharing
-2 ảnh minh họa 
-video sau khi dự đoán 
-data.yaml với 3 classes 
##  📁 File trong repo 
- `train.py` – Mã huấn luyện mô hình và dự đoán ảnh.
- `predict_video.py` – Nhận diện trong video.
- `best.pt` – Trọng số mô hình tốt nhất..
- `README.md` – Tệp hướng dẫn này.
## Những khó khăn trong quá trình huấn luyện 
-số lần huấn luyện 50 khá lâu nên chỉ huấn luyện được 25 lần 
-mới tập dùng colab và học yolov8 
-giới hạn tốc độ chạy T4 của colab 
-colab không tự lưu kết quả nên đã mất nhiều lần trainning lại 
## Các thư viện chính cần thiết 
-ultralytics	Sử dụng mô hình YOLOv8 để phân tích và dự đoán
-opencv-python	Đọc và ghi video (cv2), xử lý ảnh khung hình
-yt_dlp	Tải video YouTube với chất lượng cao nhất ( nếu lấy video từ youtube) 
## ▶️ Hướng dẫn sử dụng (Colab)
```python
!pip install ultralytics
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict("demo.mp4", save=True)
