# =============================
# 7. TẠO GIAO DIỆN KIỂM THỬ
# =============================
# Cài đặt thư viện Gradio nếu chưa có
!pip install gradio -q

import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# --- CÁC THÔNG SỐ CẦN THIẾT ---
# Tải lại mô hình đã được huấn luyện tốt nhất
try:
    model = load_model('/content/drive/MyDrive/final_model.h5')
    print("✅ Tải mô hình 'final_model.h5' thành công!")
except Exception as e:
    print(f"Lỗi: Không thể tải mô hình. Hãy chắc chắn rằng bạn đã huấn luyện và lưu file 'final_model.h5' vào Google Drive. Chi tiết lỗi: {e}")
    # Dừng thực thi nếu không tải được model
    model = None

# Danh sách các lớp hoa (PHẢI ĐÚNG THỨ TỰ NHƯ KHI HUẤN LUYỆN)
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
img_size = (64, 64)

# --- HÀM DỰ ĐOÁN ---
# Hàm này sẽ xử lý ảnh đầu vào và trả về kết quả
def predict_flower(image):
    if model is None:
        return {"Lỗi": 1.0}
        
    # 1. Tiền xử lý ảnh giống hệt như khi huấn luyện
    # Resize ảnh về đúng kích thước
    img_resized = tf.image.resize(image, img_size)
    
    # Chuẩn hóa giá trị pixel về [0, 1]
    img_normalized = img_resized / 255.0
    
    # Mở rộng chiều để tạo thành một "batch" chứa 1 ảnh duy nhất
    # Kích thước từ (64, 64, 3) -> (1, 64, 64, 3)
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # 2. Thực hiện dự đoán
    predictions = model.predict(img_batch)
    
    # 3. Xử lý kết quả đầu ra
    # predictions[0] là mảng xác suất cho các lớp, ví dụ: [0.1, 0.05, 0.8, 0.02, 0.03]
    # Tạo một dictionary để Gradio hiển thị tên lớp và xác suất tương ứng
    scores = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    
    return scores

# --- KHỞI TẠO GIAO DIỆN ---
if model is not None:
    demo = gr.Interface(
        fn=predict_flower,
        inputs=gr.Image(label="Tải ảnh hoa của bạn lên đây"),
        outputs=gr.Label(num_top_classes=3, label="Kết quả dự đoán"),
        title="🤖 Nhận dạng 5 loài hoa",
        description="Đây là giao diện để kiểm thử mô hình ANN đã được huấn luyện. Hãy tải lên ảnh của một trong 5 loài hoa: cúc họa mi (daisy), bồ công anh (dandelion), hoa hồng (roses), hoa hướng dương (sunflowers), hoặc tulip.",
        examples=[
            ["/content/dataset/daisy/5547758_eea9edfd54_n.jpg"],
            ["/content/dataset/roses/12240303_80d87f77a3_n.jpg"],
            ["/content/dataset/tulips/11242940_d53b524741.jpg"]
        ]
    )
    
    # Chạy giao diện
    demo.launch(debug=True)