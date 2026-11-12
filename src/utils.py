"""Reproducibility helpers."""
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except ImportError: pass
