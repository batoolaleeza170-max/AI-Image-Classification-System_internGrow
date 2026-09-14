import streamlit as st
from PIL import Image
import json
import os

from streamlit_webrtc import webrtc_streamer

from modules.classifies import classify_image
from modules.camera import CameraProcessor
from modules.history import save_prediction, load_history, clear_history

from modules.evaluation import (
    accuracy,
    report,
    matrix,
    CLASS_NAMES
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Image Classification System",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f5f7fb;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 25px 10px 10px 10px;
    }

    .main-header h1 {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .main-header p {
        font-size: 18px;
        color: #666;
    }

    /* Cards */
    .info-card {
        background-color: white;
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        text-align: center;
        margin-bottom: 15px;
    }

    .info-card h3 {
        margin-bottom: 5px;
        font-size: 18px;
    }

    .info-card p {
        font-size: 25px;
        font-weight: 700;
        margin: 0;
    }

    /* Section title */
    .section-title {
        font-size: 27px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* Upload box */
    .upload-box {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
    }

    /* Prediction box */
    .prediction-box {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        margin-top: 15px;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 30px 10px;
        color: #777;
        font-size: 14px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🖼️ AI Classifier")

    st.write("---")

    st.subheader("📌 Project Information")

    st.write(
        "An AI-powered image classification system "
        "using MobileNetV2 and TensorFlow."
    )

    st.write("")

    st.markdown("### 🏷️ Supported Classes")

    st.write("🐦 Birds")
    st.write("🐱 Cats")
    st.write("🐶 Dogs")
    st.write("🐴 Horses")

    st.write("---")

    st.markdown("### 🤖 Model")

    st.info("MobileNetV2")

    st.markdown("### 🧠 Framework")

    st.info("TensorFlow")

    st.write("---")

    st.caption(
        "AI Image Classification System\n"
        "Task 3 - Week 3"
    )


# =========================================================
# HEADER
# =========================================================

st.title("🤖 AI Image Classification System")

st.caption(
    "Upload an image or use your camera to classify "
    "birds, cats, dogs, and horses using MobileNetV2."
)

st.divider()

# =========================================================
# DASHBOARD OVERVIEW
# =========================================================

history = load_history()

total_predictions = len(history)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown("""
    <div class="info-card">
        <h3>🎯 Model Accuracy</h3>
    </div>
    """, unsafe_allow_html=True)

    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )


with col2:

    st.markdown("""
    <div class="info-card">
        <h3>📜 Predictions</h3>
    </div>
    """, unsafe_allow_html=True)

    st.metric(
        "Total Predictions",
        total_predictions
    )


with col3:

    st.markdown("""
    <div class="info-card">
        <h3>🏷️ Classes</h3>
    </div>
    """, unsafe_allow_html=True)

    st.metric(
        "Supported Classes",
        len(CLASS_NAMES)
    )


with col4:

    st.markdown("""
    <div class="info-card">
        <h3>🤖 Model</h3>
    </div>
    """, unsafe_allow_html=True)

    st.metric(
        "AI Model",
        "MobileNetV2"
    )


# =========================================================
# IMAGE CLASSIFICATION
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">📤 Image Classification</div>',
    unsafe_allow_html=True
)

left_col, right_col = st.columns([1, 1])


with left_col:

    st.markdown(
        '<div class="upload-box">',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Upload a JPG, JPEG or PNG image."
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


with right_col:

    if uploaded_file is None:

        st.info(
            "👆 Upload an image from the left side "
            "to start classification."
        )


# =========================================================
# IMAGE PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.write("")

    image_col, result_col = st.columns([1, 1])


    # -----------------------------------------------------
    # Uploaded Image
    # -----------------------------------------------------

    with image_col:

        st.subheader("🖼️ Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )


        classify_button = st.button(
            "🔍 Classify Image",
            use_container_width=True
        )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    with result_col:

        st.subheader("🎯 Prediction Result")

        if classify_button:

            with st.spinner("🤖 AI is analyzing the image..."):

                predicted_class, confidence, predictions = (
                    classify_image(image)
                )


            # Save prediction
            save_prediction(
                uploaded_file.name,
                predicted_class,
                confidence
            )


            st.success(
                "✅ Classification completed successfully!"
            )


            result_col1, result_col2 = st.columns(2)


            with result_col1:

                st.metric(
                    "Predicted Class",
                    predicted_class.capitalize()
                )


            with result_col2:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )


            # Probability data

            probability_data = {

                "Birds": float(predictions[0] * 100),

                "Cats": float(predictions[1] * 100),

                "Dogs": float(predictions[2] * 100),

                "Horses": float(predictions[3] * 100)
            }


            st.write("")

            st.subheader("📊 Class Probabilities")

            st.bar_chart(
                probability_data
            )


        else:

            st.info(
                "Click **Classify Image** to get "
                "the AI prediction."
            )


# =========================================================
# PREDICTION HISTORY
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">📜 Prediction History</div>',
    unsafe_allow_html=True
)

history = load_history()


if not history.empty:

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )

    history_col1, history_col2 = st.columns([3, 1])

    with history_col1:

        st.write(
            f"📌 **Total Predictions: {len(history)}**"
        )

    with history_col2:

        if st.button(
            "🗑️ Clear History",
            use_container_width=True
        ):

            clear_history()

            st.success(
                "Prediction history cleared!"
            )

            st.rerun()

else:

    st.info(
        "No prediction history yet. "
        "Classify an image to create history."
    )


# =========================================================
# MODEL EVALUATION
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">📈 Model Evaluation</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Overall Accuracy
# ---------------------------------------------------------

accuracy_col1, accuracy_col2 = st.columns([1, 2])


with accuracy_col1:

    st.metric(
        "🎯 Overall Accuracy",
        f"{accuracy * 100:.2f}%"
    )


with accuracy_col2:

    st.info(
        "Model evaluation is performed on the validation "
        "dataset using classification metrics."
    )


# ---------------------------------------------------------
# Classification Metrics
# ---------------------------------------------------------

st.subheader("🎯 Classification Metrics")


metrics_data = {

    "Class": CLASS_NAMES,

    "Precision": [
        round(
            report[class_name]["precision"],
            3
        )
        for class_name in CLASS_NAMES
    ],

    "Recall": [
        round(
            report[class_name]["recall"],
            3
        )
        for class_name in CLASS_NAMES
    ],

    "F1-Score": [
        round(
            report[class_name]["f1-score"],
            3
        )
        for class_name in CLASS_NAMES
    ]
}


st.dataframe(
    metrics_data,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------

st.subheader("🔄 Confusion Matrix")

st.write(
    "Rows represent actual classes and columns represent "
    "predicted classes."
)

st.dataframe(
    matrix,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# TRAINING PERFORMANCE
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">📊 Training Performance</div>',
    unsafe_allow_html=True
)

HISTORY_PATH = "models/training_history.json"


if os.path.exists(HISTORY_PATH):

    with open(
        HISTORY_PATH,
        "r"
    ) as f:

        training_history = json.load(f)


    # -----------------------------------------------------
    # Accuracy
    # -----------------------------------------------------

    st.subheader("📈 Accuracy")

    accuracy_data = {

        "Training Accuracy":
            training_history["accuracy"],

        "Validation Accuracy":
            training_history["val_accuracy"]
    }

    st.line_chart(
        accuracy_data
    )


    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    st.subheader("📉 Loss")

    loss_data = {

        "Training Loss":
            training_history["loss"],

        "Validation Loss":
            training_history["val_loss"]
    }

    st.line_chart(
        loss_data
    )


else:

    st.warning(
        "Training history not found. "
        "Please run train.py first."
    )


# =========================================================
# REAL-TIME CAMERA
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🎥 Real-Time Camera Detection</div>',
    unsafe_allow_html=True
)

st.write(
    "Start the camera and show a bird, cat, dog, "
    "or horse. The AI model will attempt to classify "
    "the object in real time."
)


webrtc_streamer(
    key="real-time-camera",
    video_processor_factory=CameraProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Image Classification System • TensorFlow + MobileNetV2 • 2026"
)