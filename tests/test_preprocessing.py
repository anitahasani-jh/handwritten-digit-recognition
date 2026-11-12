from PIL import Image,ImageDraw
import numpy as np
import pytest
from src.inference import preprocess_pil_image
from src.data import validate_image_batch

def test_preprocessing_shape_and_range():
    im=Image.new("L",(100,100),255); ImageDraw.Draw(im).line((45,15,45,85),fill=0,width=9)
    x=preprocess_pil_image(im)
    assert x.shape==(1,28,28,1) and x.dtype==np.float32 and 0<=x.min()<=x.max()<=1

def test_blank_image_rejected():
    with pytest.raises(ValueError,match="blank"): preprocess_pil_image(Image.new("L",(28,28),255))

def test_batch_validation():
    assert validate_image_batch(np.zeros((2,28,28))).shape==(2,28,28,1)

def test_invalid_shape_rejected():
    with pytest.raises(ValueError): validate_image_batch(np.zeros((2,32,32)))
