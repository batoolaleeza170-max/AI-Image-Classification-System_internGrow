import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# =========================
# Configuration
# =========================

MODEL_PATH = "models/image_classifier.keras"
DATASET_DIR = "data/dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 8

CLASS_NAMES = [
    "birds",
    "cats",
    "dogs",
    "horses"
]


# =========================
# Load Model
# =========================

model = tf.keras.models.load_model(MODEL_PATH)


# =========================
# Load Validation Dataset
# =========================

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
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
# Generate Predictions
# =========================

predictions = model.predict(validation_data)

predicted_labels = np.argmax(predictions, axis=1)

true_labels = validation_data.classes


# =========================
# Accuracy
# =========================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)


# =========================
# Classification Report
# =========================

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)


# =========================
# Confusion Matrix
# =========================

matrix = confusion_matrix(
    true_labels,
    predicted_labels
)