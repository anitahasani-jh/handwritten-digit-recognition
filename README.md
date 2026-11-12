# Handwritten Digit Recognition with CNN

A complete educational deep-learning project that classifies handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The repository includes data loading, model training, evaluation, custom-image prediction, a Streamlit interface, and automated tests.

## Features

- Official MNIST split: 60,000 training images and 10,000 test images.
- CNN with convolutional layers, batch normalization, max pooling, and dropout.
- Best-validation-checkpoint saving, early stopping, and learning-rate scheduling.
- Test-set loss and accuracy, per-class precision/recall/F1, and a confusion matrix.
- Single-image prediction from PNG/JPG files.
- Streamlit web interface.
- Unit tests for preprocessing and input validation.
- No fabricated benchmark numbers: run the experiment to obtain your own metrics.

## Project structure

```text
handwritten-digit-recognition/
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── inference.py
│   └── utils.py
├── tests/
│   └── test_preprocessing.py
├── artifacts/                 # Generated model and evaluation outputs
├── reports/                   # Reserved for additional reports
├── app.py
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

## Requirements

- Python 3.10–3.12 (64-bit recommended).
- Internet connection for the first MNIST download.
- A TensorFlow-supported CPU or GPU environment.

## Installation (Windows CMD)

Clone the repository and move into its directory:

```cmd
git clone https://github.com/anitahasani-jh/handwritten-digit-recognition.git
cd handwritten-digit-recognition
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.12 is not installed, replace `py -3.12` with `py`.

## Run automated tests

```cmd
pytest -q
```

These tests cover image preprocessing and input validation. They do not require downloading MNIST or training the network.

## Train

```cmd
python train.py --epochs 12 --batch-size 128
```

The MNIST dataset is downloaded automatically on the first run. The best checkpoint is saved to `artifacts/mnist_cnn.keras`, and epoch metrics are saved to `artifacts/history.json`. Training duration depends on your hardware.

Optional arguments:

- `--epochs`: maximum number of training epochs (default: 12).
- `--batch-size`: training batch size (default: 128).
- `--seed`: random seed (default: 42).
- `--validation-split`: fraction of the training partition used for validation (default: 0.1).

## Evaluate

After training, run:

```cmd
python evaluate.py
```

Evaluation uses the official MNIST test partition, which is not used for gradient updates or checkpoint selection. It creates:

- `artifacts/classification_report.txt`
- `artifacts/confusion_matrix.png`
- `artifacts/training_curves.png` (if training history is available)

The script prints the actual test loss, accuracy, and classification report. Results are intentionally not stated in advance.

## Predict a custom image

```cmd
python predict.py --image path\to\your_digit.png
```

The image should contain one clearly visible digit. Preprocessing converts it to grayscale, adjusts foreground/background polarity, crops the visible digit, resizes it while preserving aspect ratio, and centers it on a 28×28 canvas.

MNIST-like images generally work best. Photographs, multiple digits, complex backgrounds, and unusual handwriting may reduce prediction reliability.

## Launch the Streamlit app

Train the model first, then run:

```cmd
streamlit run app.py
```

Open the local URL printed by Streamlit and upload a PNG, JPG, or JPEG image containing one digit.

## Model architecture

Input: 28×28 grayscale image with one channel. The network contains two convolutional feature-extraction blocks (32 and 64 filters), batch normalization, max pooling, and dropout. A 128-unit dense layer feeds a 10-unit softmax output, representing the ten digit classes.

The model is compiled with Adam and sparse categorical cross-entropy.

## Reproducibility and limitations

- The default random seed is 42. Some GPU operations may remain nondeterministic.
- Validation data is taken from the training partition.
- The official test partition is reserved for final evaluation.
- The displayed confidence is a softmax score; it is not necessarily calibrated and is not a guarantee of correctness.
- The custom-image preprocessing is a practical approximation of MNIST formatting. It may need adjustment for unusual input styles.
- This project is intended for learning and demonstration, not high-stakes decisions.

## Dataset and references

MNIST is loaded using the Keras dataset API. See the [Keras MNIST documentation](https://keras.io/api/datasets/mnist/) and the [original MNIST database page](http://yann.lecun.com/exdb/mnist/). Dataset terms and attribution are separate from this repository's MIT-licensed source code.

## License

The source code in this repository is distributed under the MIT License. See [LICENSE](LICENSE).
