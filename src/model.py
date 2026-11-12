"""CNN architecture for MNIST."""
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape=(28, 28, 1), num_classes=10):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(), layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(), layers.Dropout(0.25),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(), layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(), layers.Dropout(0.25), layers.Flatten(),
        layers.Dense(128, activation="relu"), layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")])
    model.compile(optimizer=keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
