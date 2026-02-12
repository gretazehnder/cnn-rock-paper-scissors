from pathlib import Path
import json
import csv

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import sys

#adding project root to python path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from preprocessing.data_pipeline import get_datasets


#paths

#getting project root 
BASE_DIR = Path(__file__).resolve().parents[3]

#paths to model and evaluation folder
MODEL_DIR = BASE_DIR / "models" / "model_a"
EVAL_DIR = MODEL_DIR / "evaluation_a"

MODEL_PATH = MODEL_DIR / "model_a.keras"
HISTORY_PATH = MODEL_DIR / "history_a.json"


#utility functions

def plot_history_from_json(history_path: Path, out_dir: Path): #plotting training and validation curves from saved history

    with open(history_path, "r") as f:
        history = json.load(f)

    #accuracy plot
    plt.figure()
    plt.plot(history["accuracy"], label="Train accuracy")
    plt.plot(history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=15)
    plt.title("Model A - accuracy", fontsize=20)
    plt.savefig(out_dir / "accuracy_a.png") 
    plt.close()

    #loss plot
    plt.figure()
    plt.plot(history["loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize=15)
    plt.title("Model A - loss", fontsize=20)
    plt.savefig(out_dir / "loss_a.png") 

def collect_predictions(model, dataset):
    # lists to store true labels, predicted labels and images
    y_true = []
    y_pred = []
    x_images = []

    # loop over the dataset batch by batch
    for images, labels in dataset:

        # model predictions for the batch
        predictions = model.predict(images, verbose=0)

        # loop over images inside the batch
        for i in range(len(labels)):
            y_true.append(int(labels[i]))                 # true label
            y_pred.append(int(np.argmax(predictions[i]))) # predicted label
            x_images.append(images[i].numpy())            # image itself

    return np.array(y_true), np.array(y_pred), np.array(x_images)    

def save_confusion_matrix(y_true, y_pred, class_names, out_path):
    #computing and saving confusion matrix

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

    plt.figure()
    disp.plot(values_format="d")
    plt.title("confusion matrix - validation")
    plt.savefig(out_path)
    plt.close()


def save_misclassified_grid(x_images, y_true, y_pred, class_names, out_path, n=9):
    #saving a grid with wrong predictions

    #finding positions where the true label is different from the predicted label
    mis_idx = np.where(y_true != y_pred)[0]

    if len(mis_idx) == 0:
        return

    #taking the first n misclassified
    rows = 3
    cols = 3
    mis_idx = mis_idx[: min(n, rows * cols)]
    
    plt.figure()
    
    
    
    #looping over the selected wrong indices and placing each image in the grid
    for i, idx in enumerate(mis_idx, start=1):
        plt.subplot(rows, cols, i) #selecting the i-th cell of a rows x cols grid
        plt.imshow(x_images[idx]) #showing the misclassified image
        plt.title(
            f"true: {class_names[y_true[idx]]}\n" #converting numeric labels to readable class names
            f"pred: {class_names[y_pred[idx]]}"
            )
        plt.axis("off")#no axes
        
    plt.tight_layout(pad=2.0)

    #saving grid figure
    plt.savefig(out_path)
    plt.close()


def save_misclassified_csv(y_true, y_pred, class_names, out_path):
    #saving all misclassified samples to csv (for report)

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
    #creating evaluation directory if needed
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    #setting seed for reproducibility
    tf.keras.utils.set_random_seed(42)

    #loading trained model
    model = keras.models.load_model(MODEL_PATH)

    #(safety) ensuring the model is compiled before calling evaluate
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    #loading datasets (keeping only validation)
    _, val_ds, _ = get_datasets()

    #class names (fixed order inferred from directory names)
    class_names = ["paper", "rock", "scissors"]

    #plotting training curves
    plot_history_from_json(HISTORY_PATH, EVAL_DIR)

    #final evaluation on validation set
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"validation loss: {val_loss:.4f} | validation accuracy: {val_acc:.4f}")

    #collecting predictions
    y_true, y_pred, x_images = collect_predictions(model, val_ds)

    #confusion matrix
    save_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        EVAL_DIR / "confusion_matrix_val_a.png",
    )

    #classification report (printed and saved)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
    )
    print("\nclassification report (validation):\n")
    print(report)

    with open(EVAL_DIR / "classification_report_val_a.txt", "w") as f:
        f.write(report)

    #misclassified visualization
    save_misclassified_grid(
        x_images,
        y_true,
        y_pred,
        class_names,
        EVAL_DIR / "misclassified_val_a.png",
    )

    #misclassified csv for report
    save_misclassified_csv(
        y_true,
        y_pred,
        class_names,
        EVAL_DIR / "misclassified_val_a.csv",
    )

    print(f"\nEvaluation artifacts saved in: {EVAL_DIR}")


if __name__ == "__main__":
    main()