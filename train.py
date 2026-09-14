import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import os
import json


# =========================
# Configuration
# =========================

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10

DATASET_DIR = "data/dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "image_classifier.keras")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# Data Preparation
# =========================

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)


train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)


validation_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)


# =========================
# MobileNetV2 Base Model
# =========================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False


# =========================
# Build Model
# =========================

model = models.Sequential([
    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.4),

    layers.Dense(
        train_data.num_classes,
        activation="softmax"
    )
])


# =========================
# Compile Model
# =========================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================
# Model Summary
# =========================

model.summary()


# =========================
# Training
# =========================

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)


# =========================
# Save Training History
# =========================

with open(HISTORY_PATH, "w") as f:
    json.dump(history.history, f)


# =========================
# Save Model
# =========================

model.save(MODEL_PATH)


# =========================
# Training Completed
# =========================

print("\nTraining completed successfully!")

print(f"Model saved at: {MODEL_PATH}")

print(f"Training history saved at: {HISTORY_PATH}")

print("Class labels:", train_data.class_indices)