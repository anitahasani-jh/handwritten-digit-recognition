"""Train the CNN on MNIST."""
import argparse
from pathlib import Path
from tensorflow import keras
from src.data import load_mnist
from src.model import build_model
from src.utils import set_seed

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--epochs",type=int,default=12); p.add_argument("--batch-size",type=int,default=128); p.add_argument("--seed",type=int,default=42); p.add_argument("--validation-split",type=float,default=0.1); a=p.parse_args()
    if a.epochs<1 or a.batch_size<1: p.error("epochs and batch-size must be positive")
    if not 0.05<=a.validation_split<=0.3: p.error("validation-split must be between 0.05 and 0.30")
    set_seed(a.seed); (x,y),_=load_mnist(); model=build_model(); out=Path("artifacts"); out.mkdir(exist_ok=True)
    callbacks=[keras.callbacks.ModelCheckpoint(out/"mnist_cnn.keras",monitor="val_accuracy",save_best_only=True),keras.callbacks.EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True),keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=2,min_lr=1e-6)]
    h=model.fit(x,y,epochs=a.epochs,batch_size=a.batch_size,validation_split=a.validation_split,callbacks=callbacks)
    import json
    (out/"history.json").write_text(json.dumps({k:[float(v) for v in vals] for k,vals in h.history.items()},indent=2),encoding="utf-8")
    print("Best checkpoint:",out/"mnist_cnn.keras"); print("Run python evaluate.py for final test metrics.")

if __name__=="__main__": main()
