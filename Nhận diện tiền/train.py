# ================================================
# HUẤN LUYỆN "CHÍNH THỨC"
# ================================================

!pip install -q tensorflow gradio

import os
import shutil
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# --- BƯỚC 1: KẾT NỐI VÀ CHUẨN BỊ DỮ LIỆU ---
from google.colab import drive
drive.mount('/content/drive')

print("Sao chép dữ liệu vào máy ảo...")
source_dir = "/content/drive/MyDrive/alo/data_folder/"
local_dir = "/content/local_data_folder/"
if os.path.exists(local_dir): shutil.rmtree(local_dir)
!cp -r "{source_dir}" "{local_dir}"
print("Sao chép hoàn tất!")

# --- BƯỚC 2: TẢI VÀ XỬ LÝ ẢNH ---
data_dir = local_dir
class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
num_classes = len(class_names)
img_size = (128, 128)
print(f"Các lớp sẽ được huấn luyện: {class_names}")

X, y = [], []
for idx, cls in enumerate(class_names):
    cls_path = os.path.join(data_dir, cls)
    imgs_list = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg",".jpeg",".png"))]
    for f in tqdm(imgs_list, desc=f"Lớp {cls}"):
        try:
            img = load_img(os.path.join(cls_path, f), target_size=img_size)
            arr = img_to_array(img) / 255.0
            X.append(arr); y.append(idx)
        except: continue
X = np.array(X)
y_cat = to_categorical(np.array(y), num_classes=num_classes)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y_cat, random_state=42)

# --- BƯỚC 3: XÂY DỰNG VÀ HUẤN LUYỆN MODEL ---
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.15, horizontal_flip=True)
datagen.fit(X_train)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(img_size[0], img_size[1], 3)),
    BatchNormalization(), Conv2D(32, (3, 3), activation='relu', padding='same'), BatchNormalization(), MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(), Conv2D(64, (3, 3), activation='relu', padding='same'), BatchNormalization(), MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(), Conv2D(128, (3, 3), activation='relu', padding='same'), BatchNormalization(), MaxPooling2D((2, 2)),
    Flatten(), Dense(512, activation='relu'), Dropout(0.5), Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

save_path = "/content/drive/MyDrive/alo/best_model.h5"
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1)
]

print("\nBắt đầu quá trình huấn luyện chính thức...")
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), epochs=15, callbacks=callbacks)

print(f"\n✅ Huấn luyện chính thức hoàn tất! Model đã được lưu tại: {save_path}")