"""Custom-image preprocessing and model inference."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras

def preprocess_pil_image(image):
    if image is None: raise ValueError("No image was provided.")
    arr = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
    if float(arr.mean()) > 127: arr = 255 - arr
    mask = arr > 25
    if not mask.any(): raise ValueError("The image appears blank; upload one handwritten digit.")
    ys, xs = np.where(mask)
    crop = Image.fromarray(arr[ys.min():ys.max()+1, xs.min():xs.max()+1])
    crop.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(crop, ((28-crop.width)//2, (28-crop.height)//2))
    return (np.asarray(canvas, dtype=np.float32) / 255.0)[None, ..., None]

def load_trained_model(model_path):
    path = Path(model_path)
    if not path.is_file(): raise FileNotFoundError(f"Model not found: {path}. Run train.py first.")
    return keras.models.load_model(path)

def predict_image(model, image):
    probabilities = np.asarray(model.predict(preprocess_pil_image(image), verbose=0))[0]
    digit = int(np.argmax(probabilities))
    return digit, float(probabilities[digit]), probabilities
