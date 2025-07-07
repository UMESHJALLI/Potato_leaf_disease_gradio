import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model("potato_model.keras")

# Class labels
class_names = ['Early Blight', 'Late Blight', 'Healthy']

# Prediction function
def predict(image):
    image = image.resize((256, 256))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return f"{predicted_class} ({confidence:.2%} confidence)"

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Potato Plant Disease Detector",
    description="Upload an image of a potato leaf to detect disease."
)

interface.launch()