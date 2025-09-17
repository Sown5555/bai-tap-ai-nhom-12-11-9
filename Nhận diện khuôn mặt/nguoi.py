import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw, ImageFont

# ==== 1. Load model đã train từ Google Drive ====
model_path = "/content/drive/MyDrive/nhandien/food_cnn_doan.h5"
model = load_model(model_path)

# ==== 2. Class labels (5 loại đồ ăn) ====
class_names = ["pizza", "burger", "sushi", "salad", "pasta"]

# ==== 3. Hàm dự đoán ====
def predict_and_render(image):
    # resize ảnh về 128x128 như khi train
    img = image.resize((128,128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # dự đoán
    preds = model.predict(img_array)
    class_index = np.argmax(preds[0])
    confidence = preds[0][class_index]

    # tạo ảnh hiển thị với nhãn
    disp_img = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(disp_img)
    font = ImageFont.load_default()

    text = f"{class_names[class_index]} ({confidence*100:.2f}%)"
    text_w, text_h = draw.textbbox((0,0), text, font=font)[2:]

    # vẽ nền nhãn pastel (xanh mint nhạt)
    bg_color = (173, 216, 230, 180)
    draw.rectangle([0, 0, text_w+12, text_h+12], fill=bg_color)
    draw.text((6,6), text, font=font, fill=(0,0,0,255))

    return disp_img, f"""
    <div style='text-align:center; font-size:20px; color:#333;'>
        🍽️ Tôi dự đoán đây là:
        <span style='color:#ff7f50; font-weight:bold;'>{class_names[class_index]}</span><br>
        🔮 Độ tin cậy: <span style='color:#4682b4;'>{confidence*100:.2f}%</span>
    </div>
    """

# ==== 4. Xây dựng giao diện Gradio ====
with gr.Blocks(css="""
    .gradio-container {background: linear-gradient(to right, #fcefe6, #fefaf6);}
    #title {text-align:center; font-size:32px; color:#ff7f50; font-weight:bold;}
    #subtitle {text-align:center; font-size:18px; color:#666;}
    #predict-btn {background-color:#ffb6b9; color:white; font-size:18px; border-radius:12px;}
    #predict-btn:hover {background-color:#ff999c;}
""") as demo:
    gr.HTML("<div id='title'>🍔 Ứng dụng Nhận Diện Đồ Ăn 🍕</div>")
    gr.HTML("<div id='subtitle'>Upload ảnh và để AI dự đoán món ăn cho bạn ✨</div><br>")

    with gr.Row():
        with gr.Column(scale=6):
            img_in = gr.Image(label="📷 Upload ảnh đồ ăn", type="pil")
            btn = gr.Button("🍴 Dự đoán", elem_id="predict-btn")
        with gr.Column(scale=6):
            img_out = gr.Image(label="📊 Kết quả dự đoán", type="pil")
            text_out = gr.HTML()

    btn.click(fn=predict_and_render, inputs=img_in, outputs=[img_out, text_out])

demo.launch(debug=True, share=True)
