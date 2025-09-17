import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw, ImageFont

# ==== 1. Load model đã train ====
model_path = "/content/drive/MyDrive/nhandien/nguoi_cnn_doan.h5"
model = load_model(model_path)

# ==== 2. Class labels (tự động lấy từ train_generator.class_indices) ====
class_names = list(train_generator.class_indices.keys())
print("🔑 Các lớp nhận diện:", class_names)

# ==== 3. Hàm dự đoán ====
def predict_and_render(image):
    img = image.resize((128,128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_index = np.argmax(preds[0])
    confidence = preds[0][class_index]

    # tạo ảnh hiển thị với nhãn
    disp_img = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(disp_img)
    font = ImageFont.load_default()

    text = f"{class_names[class_index]} ({confidence*100:.2f}%)"
    text_w, text_h = draw.textbbox((0,0), text, font=font)[2:]
    bg_color = (173, 216, 230, 200)  # pastel xanh nhạt
    draw.rectangle([0, 0, text_w+10, text_h+10], fill=bg_color)
    draw.text((5,5), text, font=font, fill=(0,0,0,255))

    return disp_img, f"<h3 style='color:#2c3e50;'>👤 Đây là: <span style='color:#2980b9'>{class_names[class_index]}</span><br>🎯 Độ tin cậy: {confidence*100:.2f}%</h3>"

# ==== 4. Giao diện Gradio ====
with gr.Blocks(css=".gradio-container {background-color: #f9f9f9}") as demo:
    gr.Markdown("<h1 style='text-align:center; color:#2980b9;'>👥 Ứng dụng Nhận Diện Thành Viên Nhóm</h1>")
    gr.Markdown("<p style='text-align:center; color:#555;'>Upload ảnh khuôn mặt để hệ thống dự đoán.</p>")

    with gr.Row():
        with gr.Column(scale=6):
            img_in = gr.Image(label="📷 Upload ảnh khuôn mặt", type="pil")
            btn = gr.Button("🔍 Nhận diện", elem_id="predict-btn")
        with gr.Column(scale=6):
            img_out = gr.Image(label="Kết quả nhận diện", type="pil")
            text_out = gr.HTML()

    btn.click(fn=predict_and_render, inputs=img_in, outputs=[img_out, text_out])

demo.launch(debug=True)
