import pandas as pd
import os

HISTORY_FILE = "prediction_history.csv"


def save_prediction(image_name, predicted_class, confidence):
    """Save a prediction to history."""

    new_record = pd.DataFrame([{
        "Image": image_name,
        "Prediction": predicted_class,
        "Confidence": f"{confidence:.2f}%"
    }])

    if os.path.exists(HISTORY_FILE):
        history = pd.read_csv(HISTORY_FILE)
        history = pd.concat([history, new_record], ignore_index=True)
    else:
        history = new_record

    history.to_csv(HISTORY_FILE, index=False)


def load_history():
    """Load prediction history."""

    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)

    return pd.DataFrame(
        columns=["Image", "Prediction", "Confidence"]
    )


def clear_history():
    """Clear all prediction history."""

    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)