"""Predict one handwritten digit from an image file."""
import argparse
from PIL import Image
from src.inference import load_trained_model,predict_image

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--image",required=True); p.add_argument("--model",default="artifacts/mnist_cnn.keras"); a=p.parse_args()
    model=load_trained_model(a.model)
    with Image.open(a.image) as im: digit,confidence,_=predict_image(model,im)
    print(f"Predicted digit: {digit}"); print(f"Model confidence: {confidence:.2%}")

if __name__=="__main__": main()
