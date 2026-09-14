from PIL import Image
import numpy as np


IMG_SIZE = (224, 224)


def preprocess_image(image):
    """
    Resize and normalize an image for model prediction.
    """

    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image)
    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    return image_array