# =============================
# 1. KẾT NỐI GOOGLE DRIVE
# =============================
from google.colab import drive
drive.mount('/content/drive')

# =============================
# 2. TẢI VÀ SAO CHÉP DATASET
# =============================
import os
import shutil
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

drive_dataset = "/content/drive/MyDrive/flower_photos/flower_photos"
# Tạo một thư mục gốc để sao chép
local_base_dir = "/content/dataset"

# Kiểm tra nếu dataset đã có trên Drive thì copy qua, nếu không thì tải về
if os.path.exists(drive_dataset):
    print("📁 Sao chép dataset từ Google Drive về local...")
    if os.path.exists(local_base_dir):
        shutil.rmtree(local_base_dir)
    shutil.copytree(drive_dataset, local_base_dir)
else:
    print("🌐 Tải dataset từ TensorFlow...")
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir_path = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    # Lưu vào cả Drive và local
    shutil.copytree(data_dir_path, drive_dataset, dirs_exist_ok=True)
    shutil.copytree(data_dir_path, local_base_dir, dirs_exist_ok=True)

# !!! DÒNG QUAN TRỌNG ĐÃ ĐƯỢC SỬA !!!
# Trỏ thẳng vào thư mục chứa các lớp hoa
local_dataset = local_base_dir

print("✅ Tải và sao chép dataset thành công!")


# =============================
# 3. TIỀN XỬ LÝ DỮ LIỆU
# =============================
print("\n⏳ Bắt đầu tiền xử lý dữ liệu...")
img_size = (64, 64)
X, y = [], []

# Lấy danh sách tên các lớp (thư mục)
class_names = sorted([d for d in os.listdir(local_dataset) if os.path.isdir(os.path.join(local_dataset, d))])
print(f"Các lớp được tìm thấy: {class_names}")

# Đọc từng ảnh, chuyển thành mảng và gán nhãn
for label, folder_name in enumerate(class_names):
    folder_path = os.path.join(local_dataset, folder_name)
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        try:
            img = load_img(img_path, target_size=img_size)
            arr = img_to_array(img)
            X.append(arr)
            y.append(label)
        except Exception as e:
            # Bỏ qua các file không phải ảnh
            # print(f"Bỏ qua file {img_path} vì lỗi: {e}") # Bỏ comment nếu muốn debug
            continue

# Chuẩn hóa dữ liệu ảnh về khoảng [0, 1] và chuyển nhãn về dạng one-hot
X = np.array(X, dtype="float32") / 255.0
y = to_categorical(np.array(y), num_classes=len(class_names))

# Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Đã tìm thấy và xử lý {len(X)} ảnh.")
print(f"Kích thước tập huấn luyện (X_train): {X_train.shape}")
print(f"Kích thước tập kiểm thử (X_test): {X_test.shape}")
print("✅ Tiền xử lý dữ liệu hoàn tất!")

# =============================
# 4. DATA AUGMENTATION (TĂNG CƯỜNG DỮ LIỆU)
# =============================
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)
print("\n⚙️  Đã cấu hình Data Augmentation.")

# =============================
# 5. XÂY DỰNG MÔ HÌNH ANN
# =============================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

model = Sequential([
    Flatten(input_shape=(64, 64, 3)),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint('/content/drive/MyDrive/final_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
]
print("✅ Xây dựng mô hình thành công!")

# =============================
# 6. HUẤN LUYỆN MÔ HÌNH
# =============================
print("\n🚀 Bắt đầu huấn luyện mô hình...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=25,
    callbacks=callbacks
)

print("\n🎉 Đã huấn luyện xong và lưu mô hình tốt nhất vào Google Drive: final_model.h5")