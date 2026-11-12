"""Evaluate the saved model on the untouched MNIST test split."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.data import load_mnist
from src.inference import load_trained_model

def main():
    out=Path("artifacts"); out.mkdir(exist_ok=True); model=load_trained_model(out/"mnist_cnn.keras"); (_, _),(x,y)=load_mnist()
    loss,acc=model.evaluate(x,y,verbose=0); pred=np.argmax(model.predict(x,batch_size=256,verbose=0),axis=1)
    report=classification_report(y,pred,labels=list(range(10)),zero_division=0); (out/"classification_report.txt").write_text(report,encoding="utf-8")
    ConfusionMatrixDisplay(confusion_matrix(y,pred,labels=list(range(10))),display_labels=list(range(10))).plot(cmap="Blues",values_format="d"); plt.tight_layout(); plt.savefig(out/"confusion_matrix.png",dpi=160); plt.close()
    history=out/"history.json"
    if history.exists():
        import json
        h=json.loads(history.read_text(encoding="utf-8"))
        if "accuracy" in h and "val_accuracy" in h:
            plt.plot(h["accuracy"],label="Training"); plt.plot(h["val_accuracy"],label="Validation"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout(); plt.savefig(out/"training_curves.png",dpi=160); plt.close()
    print(f"Test loss: {loss:.4f}"); print(f"Test accuracy: {acc:.4%}"); print(report)

if __name__=="__main__": main()
