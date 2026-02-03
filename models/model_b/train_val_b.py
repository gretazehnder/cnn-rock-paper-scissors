from pathlib import Path
import json
import csv
import itertools
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sys

#adding project root to python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from preprocessing.data_pipeline import get_datasets, get_augmentation_layer, IMAGE_SIZE

#project root directory
BASE_DIR = PROJECT_ROOT

#output directory for model b artifacts
OUT_DIR = BASE_DIR / "models" / "model_b"

NUM_CLASSES = 3


#function for model b (tunable)
def build_model_b(
    augmentation_layer,
    n_blocks: int,
    base_filters: int,
    dense_units: int,
    dropout_conv: float,
    dropout_dense: float,
    use_batchnorm: bool,
):
   
    model = keras.Sequential(name="model_b_tuned")
    
    model.add(layers.Input(shape=(*IMAGE_SIZE, 3), name="input_image"))
    model.add(augmentation_layer)

    filters = base_filters
    for b in range(n_blocks):
        #conv block
        model.add(
            layers.Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                padding="same",
                use_bias=not use_batchnorm,
            )
        )
        if use_batchnorm:
            model.add(layers.BatchNormalization())
            
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        if dropout_conv > 0:
            model.add(layers.Dropout(dropout_conv))

        filters *= 2  #increasing depth each block

    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation="relu"))

    if dropout_dense > 0:
        model.add(layers.Dropout(dropout_dense))

    model.add(layers.Dense(NUM_CLASSES, activation="softmax", name="pred"))
    
    return model

#saving hyperparameter tuning results (one row per configuration) to CSV.
def save_results_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():

    #datasets (same as model A)
    train_ds, val_ds, _ = get_datasets() 
    aug = get_augmentation_layer()

    #grid search space (restricted, explicit trade-offs)
    grid = {
        "n_blocks": [2, 3], #depth vs overfitting/compute
        "base_filters": [32, 64], #capacity vs speed
        "dense_units": [128, 256], #capacity vs overfitting
        "dropout_conv": [0.0, 0.2, 0.3], #regularization vs underfitting
        "dropout_dense": [0.2, 0.4], #regularization vs underfitting
        "use_batchnorm": [True], #stability (kept fixed, still explicit)
        "learning_rate": [1e-3, 3e-4], #speed vs stability
        "epochs": [15], #fixed to compare
    } #48 configurations

    #for model b (intermediate), we do NOT repeat runs to reduce computational cost
    EARLY_STOP_PATIENCE = 3
    
    #from dictionary to a list
    keys = list(grid.keys())
    configs = list(itertools.product(*[grid[k] for k in keys])) #cartesian product

    results = []

    #back to dictionary
    for cfg in configs:
        cfg_dict = dict(zip(keys, cfg))

        #using a fixed seed for a fair comparison across configurations
        tf.keras.utils.set_random_seed(42)

        model = build_model_b(
            augmentation_layer=aug,
            n_blocks=int(cfg_dict["n_blocks"]),
            base_filters=int(cfg_dict["base_filters"]),
            dense_units=int(cfg_dict["dense_units"]),
            dropout_conv=float(cfg_dict["dropout_conv"]),
            dropout_dense=float(cfg_dict["dropout_dense"]),
            use_batchnorm=bool(cfg_dict["use_batchnorm"]),
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
            epochs=int(cfg_dict["epochs"]),
            validation_data=val_ds,
            callbacks=[early_stop],
            verbose=0,
        )

        best_val_acc = float(np.max(history.history["val_accuracy"]))
        best_val_loss = float(np.min(history.history["val_loss"]))
        
        #new dictionary
        row = {
            **cfg_dict,
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
        }
        results.append(row)

        print(
            f"[GRID] {cfg_dict} -> "
            f"best_val_acc={row['best_val_acc']:.4f}"
        )


    #selecting best: highest accuracy, lowest loss
    results_sorted = sorted(
        results,
        key=lambda r: (r["best_val_acc"], -r["best_val_loss"]),
        reverse=True,
    )
    best = results_sorted[0]

    #saving full results (for report)
    with open(OUT_DIR / "cv_results_b.json", "w") as f:
        json.dump(results_sorted, f, indent=2)
    save_results_csv(OUT_DIR / "cv_results_b.csv", results_sorted)

    #saving best hyperparameters
    with open(OUT_DIR / "best_hp_b.json", "w") as f:
        json.dump(best, f, indent=2)

    print("\nBest configuration (best val_accuracy, tie-break val_loss):")
    print(json.dumps(best, indent=2))


    #training final model with best HP (same style as Model A)
    tf.keras.utils.set_random_seed(42)

    model = build_model_b(
        augmentation_layer=aug,
        n_blocks=int(best["n_blocks"]),
        base_filters=int(best["base_filters"]),
        dense_units=int(best["dense_units"]),
        dropout_conv=float(best["dropout_conv"]),
        dropout_dense=float(best["dropout_dense"]),
        use_batchnorm=bool(best["use_batchnorm"]),
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
        epochs=int(best["epochs"]),
        validation_data=val_ds,
        callbacks=[early_stop],
        verbose=1,
    )

    #saving training metrics history for later analysis and plotting
    with open(OUT_DIR / "history_b.json", "w") as f:
        json.dump(history.history, f)

    #saving model
    model.save(OUT_DIR / "model_b.keras")

    print("Saved model, history, and tuning results in models/model_b/")


if __name__ == "__main__":
    main()
