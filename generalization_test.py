from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


#config
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32

#same order
CLASS_NAMES = ["paper", "rock", "scissors"]

#external dataset
GEN_DIR = Path.home() / "OneDrive" / "Desktop" / "generalization_dataset"

#model c path
MODEL_PATH = Path("models/model_c/model_c.keras") #here model c was loaded, but it is possible to load any model by changing this path

#data loading
def load_generalization_ds(gen_dir: Path) -> tf.data.Dataset:
    
    if not gen_dir.exists():
        raise FileNotFoundError(f"Generalization dataset folder not found: {gen_dir}")

    ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=gen_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,  #to allineate y_true and pred
        verbose=True,
    )

    #same normalization
    rescale = tf.keras.layers.Rescaling(1.0 / 255)
    ds = ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    #performance
    ds = ds.cache().prefetch(tf.data.AUTOTUNE)
    return ds


#evaluation
def main():
    print(f"Loading model from: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"\nLoading generalization dataset from: {GEN_DIR}")
    gen_ds = load_generalization_ds(GEN_DIR)

    #loss + accuracy
    loss, acc = model.evaluate(gen_ds, verbose=1)
    print(f"\nGeneralization results -> loss: {loss:.4f} | accuracy: {acc:.4f}")

    #predictions
    y_true = np.concatenate([y.numpy() for _, y in gen_ds])
    y_prob = model.predict(gen_ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    #classification report
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))

    #confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(values_format="d")
    plt.title(f"Model {MODEL_PATH.stem} - Generalization test (Confusion Matrix)")
    plt.tight_layout()
    plt.show()

    #quick error summary
    wrong_idx = np.where(y_true != y_pred)[0]
    print(f"\nMisclassified: {len(wrong_idx)} / {len(y_true)}")


if __name__ == "__main__":
    main()
