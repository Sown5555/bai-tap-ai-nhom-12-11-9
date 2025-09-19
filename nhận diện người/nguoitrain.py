# ======================
# 1. Import
# ======================
import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ======================
# 2. Load model chỉ tay
# ======================
hand_model_path = "/content/drive/MyDrive/nhandien/chitay_cnn_doan.h5"
hand_model = load_model(hand_model_path)

# Lấy tên class từ thư mục train (ví dụ 5 loại vân tay)
hand_classes = ['Loai1', 'Loai2', 'Loai3', 'Loai4', 'Loai5']  # đổi theo dataset của bạn

# ======================
# 3. Hàm dự đoán
# ======================
def predict_hand(img):
    img = img.convert("RGB")
    img = img.resize((128,128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    preds = hand_model.predict(x)[0]
    class_idx = np.argmax(preds)
    confidence = preds[class_idx]

    return {hand_classes[i]: float(preds[i]) for i in range(len(hand_classes))}

# ======================
# 4. Giao diện Gradio
# ======================
hand_app = gr.Interface(
    fn=predict_hand,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="🖐️ Nhận diện Chỉ Tay",
    description="Tải ảnh chỉ tay để phân loại loại vân tay."
)

hand_app.launch(share=True)
