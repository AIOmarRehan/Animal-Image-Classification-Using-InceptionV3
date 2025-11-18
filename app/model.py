import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained CNN model
# In the notebook, the file was named “Simple_CNN_Classification.hs.” You may rename it locally on your machine as needed
model = tf.keras.models.load_model("saved_model/Animal_Classification.h5")

# Same label order you used when training (from LabelEncoder)
CLASS_NAMES = ["Cat", "Dog", "Snake"]

def preprocess_image(img: Image.Image, target_size=(256, 256)):
    img = img.convert("RGB")  # ensure 3 channels
    img = img.resize(target_size)
    img = np.array(img).astype("float32") / 255.0  # normalize
    img = np.expand_dims(img, axis=0)  # (1, 256, 256, 3)
    return img

def predict(img: Image.Image):
    # Apply preprocessing
    input_tensor = preprocess_image(img)

    # Model prediction
    preds = model.predict(input_tensor)
    probs = preds[0]

    class_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))

    # Map all probabilities
    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    return CLASS_NAMES[class_idx], confidence, prob_dict

