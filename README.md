# 🖼️ AI Image Classification System

An AI-powered image classification system built using **Python, TensorFlow, Keras, MobileNetV2 and Streamlit**.

The system can classify images into four categories — **Birds, Cats, Dogs, and Horses** — and provides prediction confidence, classification history, model evaluation, training performance graphs, and real-time camera detection.

---

## 🚀 Features

### 📤 Image Upload & Classification
- Upload JPG, JPEG, or PNG images.
- Classify uploaded images using a trained MobileNetV2 model.
- Display the predicted class.
- Display prediction confidence.

### 📊 Prediction Dashboard
- Class probability visualization.
- Overall model accuracy.
- Total prediction count.
- Supported class information.
- AI model information.

### 📜 Prediction History
- Automatically saves image predictions.
- Stores predicted class and confidence.
- Displays previous predictions in a table.
- Option to clear prediction history.

### 📈 Model Evaluation
- Overall accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.

### 📉 Training Performance
- Training accuracy graph.
- Validation accuracy graph.
- Training loss graph.
- Validation loss graph.

### 🎥 Real-Time Camera Detection
- Uses the device camera for real-time classification.
- Displays the predicted class and confidence directly on the camera feed.

---

## 🧠 Supported Classes

The model currently supports four classes:

| Class | Label |
|---|---|
| 🐦 Birds | Birds |
| 🐱 Cats | Cats |
| 🐶 Dogs | Dogs |
| 🐴 Horses | Horses |

---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow**
- **Keras**
- **MobileNetV2**
- **OpenCV**
- **Streamlit**
- **Streamlit-WebRTC**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **Pillow**

---

## 🏗️ Project Structure

```text
AI-Image-Classification-System/
│
├── data/
│   └── dataset/
│       ├── birds/
│       ├── cats/
│       ├── dogs/
│       └── horses/
│
├── models/
│   ├── image_classifier.keras
│   └── training_history.json
│
├── modules/
│   ├── classifies.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   ├── history.py
│   └── camera.py
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
