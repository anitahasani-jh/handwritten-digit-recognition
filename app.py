"""Streamlit demo for the trained MNIST CNN."""
from pathlib import Path
import streamlit as st
from PIL import Image
from src.inference import load_trained_model,predict_image

st.set_page_config(page_title="Handwritten Digit Recognition",page_icon="✍️")
st.title("Handwritten Digit Recognition")
st.write("Upload one handwritten digit (0–9).")
path=Path("artifacts/mnist_cnn.keras")
if not path.exists(): st.warning("Train the model first: python train.py"); st.stop()
@st.cache_resource
def get_model(p): return load_trained_model(p)
file=st.file_uploader("Choose a PNG or JPG image",type=["png","jpg","jpeg"])
if file:
    try:
        image=Image.open(file).convert("RGB"); st.image(image,caption="Uploaded image",width=220)
        digit,confidence,probs=predict_image(get_model(str(path)),image)
        st.metric("Predicted digit",str(digit)); st.metric("Model confidence",f"{confidence:.2%}")
        st.bar_chart({str(i):float(v) for i,v in enumerate(probs)})
        st.caption("Confidence is the model softmax score, not a guarantee of correctness.")
    except (ValueError,OSError) as e: st.error(str(e))
