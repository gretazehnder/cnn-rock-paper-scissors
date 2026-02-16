from pathlib import Path
import json
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#adding project root to python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from preprocessing.data_pipeline import get_datasets, get_augmentation_layer, IMAGE_SIZE

BASE_DIR = PROJECT_ROOT
OUT_DIR = BASE_DIR / "models" / "model_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 3


#function for model a (baseline)
def build_model_a(augmentation_layer):
    model = keras.Sequential(
        [
            layers.Input(shape=(*IMAGE_SIZE, 3), name="input_image"),

            #data augmentation layer
            augmentation_layer,

            #first conv block
            layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
            ),
            layers.MaxPooling2D(pool_size=(2, 2)),

            #second conv block
            layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
            ),
            layers.MaxPooling2D(pool_size=(2, 2)),

            #classifier head
            layers.Flatten(),
            layers.Dense(units=128, activation="relu"),

            #output
            layers.Dense(units=NUM_CLASSES, activation="softmax", name="pred"),
        ],
        name="model_a_baseline",
    )
    return model


def main():
    tf.keras.utils.set_random_seed(42)

    train_ds, val_ds, _ = get_datasets()
    aug = get_augmentation_layer()

    model = build_model_a(aug)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    model.summary()

    history = model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
    )

    with open(OUT_DIR / "history_a.json", "w") as f:
        json.dump(history.history, f)

    model.save(OUT_DIR / "model_a.keras")

    print("Saved model and history in models/model_a/")


if __name__ == "__main__":
    main()
