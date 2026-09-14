import cv2
import av
import numpy as np
import tensorflow as tf
from streamlit_webrtc import VideoProcessorBase

# =========================
# Model Configuration
# =========================

MODEL_PATH = "models/image_classifier.keras"

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
# Camera Processor
# =========================

class CameraProcessor(VideoProcessorBase):

    def recv(self, frame):

        # Convert camera frame to OpenCV
        img = frame.to_ndarray(format="bgr24")

        # Convert BGR to RGB
        image = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # Resize
        image = cv2.resize(
            image,
            (224, 224)
        )

        # Normalize
        image = image / 255.0

        # Add batch dimension
        image = np.expand_dims(
            image,
            axis=0
        )

        # Predict
        predictions = model.predict(
            image,
            verbose=0
        )

        # Get predicted class
        predicted_index = np.argmax(
            predictions[0]
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = (
            predictions[0][predicted_index] * 100
        )

        # Text
        text = (
            f"{predicted_class.capitalize()} "
            f"- {confidence:.2f}%"
        )

        # Display prediction
        cv2.putText(
            img,
            text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Return processed frame
        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )