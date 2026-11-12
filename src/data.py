"""MNIST data loading and input validation."""
import numpy as np
from tensorflow import keras

def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train.astype("float32") / 255.0, -1)
    x_test = np.expand_dims(x_test.astype("float32") / 255.0, -1)
    return (x_train, y_train), (x_test, y_test)

def validate_image_batch(images):
    x = np.asarray(images, dtype=np.float32)
    if x.ndim == 3: x = x[..., np.newaxis]
    if x.ndim != 4 or x.shape[1:] != (28, 28, 1):
        raise ValueError("Expected shape (N, 28, 28) or (N, 28, 28, 1).")
    if not np.isfinite(x).all(): raise ValueError("Images must be finite.")
    if x.size and (x.min() < 0 or x.max() > 255): raise ValueError("Pixel values must be in [0, 255].")
    if x.size and x.max() > 1: x = x / 255.0
    return x
