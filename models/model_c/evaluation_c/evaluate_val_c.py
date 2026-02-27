from pathlib import Path
import json
import csv

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from preprocessing.data_pipeline import get_datasets, CLASS_NAMES

MODEL_DIR = PROJECT_ROOT / "models" / "model_c"
EVAL_DIR = MODEL_DIR / "evaluation_c"

MODEL_PATH = MODEL_DIR / "model_c.keras"
HISTORY_PATH = MODEL_DIR / "history_c.json"


#utility functions

def plot_history_from_json(history_path: Path, out_dir: Path): 
    """Plot training history from json file and save the plots"""

    with open(history_path, "r") as f:
        history = json.load(f)

    #accuracy plot
    plt.figure()
    plt.plot(history["accuracy"], label="Train accuracy")
    plt.plot(history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=15)
    plt.title("Model C - accuracy", fontsize=20)
    plt.savefig(out_dir / "accuracy_c.png") 
    plt.close()

    #loss plot
    plt.figure()
    plt.plot(history["loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize=15)
    plt.title("Model C - loss", fontsize=20)
    plt.savefig(out_dir / "loss_c.png") 
    plt.close()

def collect_predictions(model, dataset):
    """Collect true labels, predicted labels and input images from the dataset"""
    y_true = []
    y_pred = []
    x_images = []

    for images, labels in dataset:

        predictions = model.predict(images, verbose=0)

        for i in range(len(labels)):
            y_true.append(int(labels[i]))                 
            y_pred.append(int(np.argmax(predictions[i]))) 
            x_images.append(images[i].numpy())            

    return np.array(y_true), np.array(y_pred), np.array(x_images)    

def save_confusion_matrix(y_true, y_pred, class_names, out_path):
    """Compute and save confusion matrix"""

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

    plt.figure()
    disp.plot(values_format="d")
    plt.title("Confusion matrix - validation")
    plt.savefig(out_path)
    plt.close()


def save_misclassified_grid(x_images, y_true, y_pred, class_names, out_path, n=9):
    """Save a grid of misclassified images"""

    mis_idx = np.where(y_true != y_pred)[0]

    if len(mis_idx) == 0:
        return

    rows = 3
    cols = 3
    mis_idx = mis_idx[: min(n, rows * cols)]
    
    plt.figure()
    
    for i, idx in enumerate(mis_idx, start=1):
        plt.subplot(rows, cols, i) 
        plt.imshow(x_images[idx]) 
        plt.title(
            f"true: {class_names[y_true[idx]]}\n" 
            f"pred: {class_names[y_pred[idx]]}"
            )
        plt.axis("off")
        
    plt.tight_layout(pad=2.0)

    plt.savefig(out_path)
    plt.close()


def save_misclassified_csv(y_true, y_pred, class_names, out_path):
    """Save all misclassified samples to csv"""

    mis_idx = np.where(y_true != y_pred)[0]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_label", "predicted_label"])

        for idx in mis_idx:
            writer.writerow([
                int(idx),
                class_names[int(y_true[idx])],
                class_names[int(y_pred[idx])],
            ])


#main function
def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(42)

    model = keras.models.load_model(MODEL_PATH)

    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    _, val_ds, _ = get_datasets()

    plot_history_from_json(HISTORY_PATH, EVAL_DIR)

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"validation loss: {val_loss:.4f} | validation accuracy: {val_acc:.4f}")

    y_true, y_pred, x_images = collect_predictions(model, val_ds)

    #confusion matrix
    save_confusion_matrix(
        y_true,
        y_pred,
        CLASS_NAMES,
        EVAL_DIR / "confusion_matrix_val_c.png",
    )

    #classification report (printed and saved)
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )
    print("\nclassification report (validation):\n")
    print(report)

    with open(EVAL_DIR / "classification_report_val_c.txt", "w") as f:
        f.write(report)

    #misclassified visualization
    save_misclassified_grid(
        x_images,
        y_true,
        y_pred,
        CLASS_NAMES,
        EVAL_DIR / "misclassified_val_c.png",
    )

    #misclassified csv for report
    save_misclassified_csv(
        y_true,
        y_pred,
        CLASS_NAMES,
        EVAL_DIR / "misclassified_val_c.csv",
    )

    print(f"\nEvaluation artifacts saved in: {EVAL_DIR}")


if __name__ == "__main__":
    main()