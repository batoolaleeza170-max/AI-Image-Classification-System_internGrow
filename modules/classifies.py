import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = "models/image_classifier.keras"

CLASS_NAMES = [
    "birds",
    "cats",
    "dogs",
    "horses"
]


# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)


def classify_image(image):
    """
    Classify an uploaded image and return
    predicted class and confidence.
    """

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image
    image = image.resize((224, 224))

    # Convert to NumPy array
    image_array = np.array(image)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(image_array, verbose=0)

    # Get highest probability
    predicted_index = np.argmax(predictions[0])

    confidence = predictions[0][predicted_index] * 100

    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence, predictions[0]