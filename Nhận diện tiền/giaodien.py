# ================================================
# KHỞI CHẠY GIAO DIỆN
# ================================================

# 1. Cài đặt các thư viện cần thiết
!pip install -q gradio tensorflow

import numpy as np
import gradio as gr
import re
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


# 2. Kết nối và tải các công cụ cần thiết
from google.colab import drive
drive.mount('/content/drive')

# Tải model từ Drive vào máy ảo để chạy nhanh hơn
drive_model_path = "/content/drive/MyDrive/alo/best_model.h5"
local_model_path = "/content/best_model.h5"
!cp "{drive_model_path}" "{local_model_path}"

# Các thông số của model
class_names = ['1000', '10000', '100000', '2000', '20000', '200000', '500', '5000', '50000', '500000']
img_size = (128, 128)

# Tải model nhận dạng hình ảnh
print("Đang tải bộ não nhận diện hình ảnh...")
best_model = load_model(local_model_path)

# 3. Tạo và chạy giao diện
iface = gr.Interface(
    fn=predict_fn,
    inputs=gr.Image(type="pil", label="Ảnh tờ tiền"),
    outputs=gr.Label(num_top_classes=3, label="Kết quả"),
    title="Nhận diện mệnh giá tiền",
    description="Tải ảnh lên để hệ thống phân tích và nhận diện."
)

print("\nĐang khởi chạy giaoG diện, vui lòng chờ...")
iface.launch(share=True)