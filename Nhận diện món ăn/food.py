from google.colab import drive
drive.mount('/content/drive')

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ======================
# ĐƯỜNG DẪN MỚI
# ======================
data_dir = '/content/drive/MyDrive/nhandien/nhandiendoan'

img_size = (128,128)
batch_size = 32   # nhỏ lại để tránh out of memory

# ======================
# DATA AUGMENTATION
# ======================
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Training generator
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Validation generator
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# ======================
# CNN MODEL
# ======================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(train_generator.num_classes, activation='softmax')
])

model.summary()

# ======================
# COMPILE MODEL
# ======================
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ======================
# TRAIN MODEL
# ======================
epochs = 30
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# ======================
# LƯU MODEL
# ======================
save_path = "/content/drive/MyDrive/nhandien/food_cnn_doan.h5"
model.save(save_path)
print("✅ Đã train xong, model lưu tại:", save_path)
