from pathlib import Path
import json
import csv
import itertools
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from preprocessing.data_pipeline import get_datasets, get_augmentation_layer, IMAGE_SIZE

BASE_DIR = PROJECT_ROOT
OUT_DIR = BASE_DIR / "models" / "model_b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 3


def build_model_b(
    augmentation_layer,
    n_blocks: int,
    base_filters: int,
    dense_units: int,
) -> keras.Model:
    model = keras.Sequential(name="model_b_tuned")

    model.add(layers.Input(shape=(*IMAGE_SIZE, 3), name="input_image"))
    model.add(augmentation_layer)

    filters = base_filters
    for _ in range(n_blocks):
        model.add(
            layers.Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            )
        )
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        filters *= 2

    # FLATTEN instead of GAP
    model.add(layers.Flatten())

    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax", name="pred"))

    return model


def save_results_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    # fixed hyperparameters
    DENSE_UNITS = 256  #larger head than Model A to accommodate the increased backbone capacity
    EPOCHS = 20
    EARLY_STOP_PATIENCE = 3

    # datasets
    train_ds, val_ds, _ = get_datasets()

    # grid search space
    grid = {
        "n_blocks": [2, 3],
        "base_filters": [32, 64],
        "learning_rate": [1e-3, 3e-4],
    }

    keys = list(grid.keys())
    configs = list(itertools.product(*[grid[k] for k in keys]))

    results = []

    for cfg in configs:
        keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)

        cfg_dict = dict(zip(keys, cfg))

        aug = get_augmentation_layer()

        model = build_model_b(
            augmentation_layer=aug,
            n_blocks=int(cfg_dict["n_blocks"]),
            base_filters=int(cfg_dict["base_filters"]),
            dense_units=DENSE_UNITS,
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=float(cfg_dict["learning_rate"])),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
        )

        history = model.fit(
            train_ds,
            epochs=EPOCHS,
            validation_data=val_ds,
            callbacks=[early_stop],
            verbose=0,
        )

        best_val_acc = float(np.max(history.history["val_accuracy"]))
        best_val_loss = float(np.min(history.history["val_loss"]))

        row = {
            **cfg_dict,
            "dense_units": DENSE_UNITS,
            "epochs": EPOCHS,
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
        }
        results.append(row)

        print(f"[GRID] {cfg_dict} -> best_val_acc={best_val_acc:.4f}")

    # select best: highest accuracy, tie-break lowest loss
    results_sorted = sorted(results, key=lambda r: (-r["best_val_acc"], r["best_val_loss"]))
    best = results_sorted[0]

    # save tuning results
    with open(OUT_DIR / "gs_results_b.json", "w") as f:
        json.dump(results_sorted, f, indent=2)
    save_results_csv(OUT_DIR / "gs_results_b.csv", results_sorted)

    with open(OUT_DIR / "best_hp_b.json", "w") as f:
        json.dump(best, f, indent=2)

    print("\nBest configuration (best val_accuracy, tie-break val_loss):")
    print(json.dumps(best, indent=2))

    # final training with best hp
    keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    aug = get_augmentation_layer()

    model = build_model_b(
        augmentation_layer=aug,
        n_blocks=int(best["n_blocks"]),
        base_filters=int(best["base_filters"]),
        dense_units=DENSE_UNITS,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(best["learning_rate"])),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
    )

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[early_stop],
        verbose=1,
    )

    with open(OUT_DIR / "history_b.json", "w") as f:
        json.dump(history.history, f)

    model.save(OUT_DIR / "model_b.keras")

    print("Saved model, history, and tuning results in models/model_b/")


if __name__ == "__main__":
    main()

