# Cài đặt thư viện cần thiết
!pip install gradio --quiet

import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- PHẦN CẤU HÌNH ---
# 1. Đường dẫn đến file model .h5 của bạn
MODEL_PATH = "/content/drive/MyDrive/nhandien/food_cnn_doan.h5"

# 2. Danh sách tên các món ăn (đã cập nhật và sắp xếp)
# QUAN TRỌNG: Thứ tự tên phải khớp với thứ tự alphabet của thư mục ảnh
CLASS_NAMES = ['burger', 'pasta', 'pizza', 'salad', 'sushi']
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
        prediction = model.predict(img_array, verbose=0) # Thêm verbose=0 để ẩn log dự đoán
        confidences = {CLASS_NAMES[i]: float(score) for i, score in enumerate(prediction[0])}
        return confidences

    # Tạo và khởi chạy giao diện
    iface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil", label="Tải ảnh món ăn của bạn lên đây"),
        outputs=gr.Label(num_top_classes=5, label="Kết quả nhận diện"),
        title="🍜 Ứng dụng Nhận diện Món ăn",
        description="Giao diện sử dụng mô hình CNN để nhận diện 5 món ăn: Burger, Pasta, Pizza, Salad, và Sushi.",
    )
    iface.launch(debug=True)
