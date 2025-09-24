# Cài đặt thư viện cần thiết
!pip install gradio --quiet

import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- PHẦN CẤU HÌNH ---
# 1. Đường dẫn đến file model .h5 của bạn trên Google Drive
MODEL_PATH = "/content/drive/MyDrive/nhandien/nguoi_cnn_doan.h5"

# 2. Danh sách tên các lớp (người)
# QUAN TRỌNG: Thứ tự tên phải khớp với thứ tự alphabet của thư mục ảnh
CLASS_NAMES = ['Nguyen Dong Son', 'Nguyen Hoang Phuc', 'Nguyen Sy Khang']
# --- KẾT THÚC CẤU HÌNH ---

# Tải model và định nghĩa hàm dự đoán
if not os.path.exists(MODEL_PATH):
    print(f"Lỗi: Không tìm thấy file model tại '{MODEL_PATH}'.")
else:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Tải model thành công!")

    def predict_image(pil_image):
        """Xử lý ảnh và trả về kết quả dự đoán."""
        if pil_image is None:
            return None

        img = pil_image.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        confidences = {CLASS_NAMES[i]: float(score) for i, score in enumerate(prediction[0])}
        return confidences

    # Tạo và khởi chạy giao diện
    iface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil", label="Tải ảnh của bạn lên đây"),
        outputs=gr.Label(num_top_classes=3, label="Kết quả nhận diện"),
        title="👤 Ứng dụng Nhận diện Người",
        description="Giao diện sử dụng mô hình CNN để nhận diện người trong ảnh.",
    )
    iface.launch(debug=True)