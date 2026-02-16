from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import sys

#adding project root to python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from preprocessing.data_pipeline import get_datasets


#paths

#project root directory
BASE_DIR = PROJECT_ROOT

MODEL_DIR = BASE_DIR / "models" / "model_c"
EVAL_DIR = MODEL_DIR / "evaluation_c"

MODEL_PATH = MODEL_DIR / "model_c.keras"


#utility functions
def collect_predictions(model, dataset):
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


def save_confusion_matrix(y_true, y_pred, class_names, out_path, title="Confusion matrix - TEST"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

    plt.figure()
    disp.plot(values_format="d")
    plt.title(title)
    plt.savefig(out_path)
    plt.close()


def save_misclassified_grid(x_images, y_true, y_pred, class_names, out_path, n=9):
    mis_idx = np.where(y_true != y_pred)[0]
    if len(mis_idx) == 0:
        return

    rows, cols = 3, 3
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


def save_test_metrics_txt(test_loss, test_acc, out_path):
    with open(out_path, "w") as f:
        f.write(f"test_loss: {test_loss:.6f}\n")
        f.write(f"test_accuracy: {test_acc:.6f}\n")


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(42)

    #loading model
    model = keras.models.load_model(MODEL_PATH)

    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    #loading test dataset
    _, _, test_ds = get_datasets()

    class_names = ["paper", "rock", "scissors"]

    #evaluation
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"TEST loss: {test_loss:.4f} | TEST accuracy: {test_acc:.4f}")

    save_test_metrics_txt(test_loss, test_acc, EVAL_DIR / "metrics_test_c.txt")

    #predictions
    y_true, y_pred, x_images = collect_predictions(model, test_ds)

    #confusion matrix
    save_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        EVAL_DIR / "confusion_matrix_test_c.png",
        title="Confusion Matrix - TEST (Model C)",
    )

    #classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
    )

    print("\nClassification report (TEST):\n")
    print(report)

    with open(EVAL_DIR / "classification_report_test_c.txt", "w") as f:
        f.write(report)

    #misclassified examples
    save_misclassified_grid(
        x_images,
        y_true,
        y_pred,
        class_names,
        EVAL_DIR / "misclassified_test_c.png",
    )

    save_misclassified_csv(
        y_true,
        y_pred,
        class_names,
        EVAL_DIR / "misclassified_test_c.csv",
    )

    print(f"\nTest evaluation artifacts saved in: {EVAL_DIR}")


if __name__ == "__main__":
    main()
