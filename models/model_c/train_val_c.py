from pathlib import Path
import json
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from preprocessing.data_pipeline import get_datasets, get_augmentation_layer, IMAGE_SIZE

BASE_DIR = PROJECT_ROOT
OUT_DIR = BASE_DIR / "models" / "model_c"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 3


def save_results_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sample_loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    log_low, log_high = np.log10(low), np.log10(high)
    return float(10 ** rng.uniform(log_low, log_high))


def build_model_c(
    augmentation_layer,
    n_blocks: int,
    base_filters: int,
    dense_units: int,
    spatial_dropout: float,
    dropout_head: float,
    l2_reg: float,
    use_batchnorm: bool,
):
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="input_image")
    x = augmentation_layer(inputs)

    filters = base_filters
    for b in range(n_blocks):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            padding="same",
            use_bias=not use_batchnorm,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(l2_reg) if l2_reg > 0 else None,
            name=f"conv_{b+1}",
        )(x)

        if use_batchnorm:
            x = layers.BatchNormalization(name=f"bn_{b+1}")(x)

        x = layers.ReLU(name=f"relu_{b+1}")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name=f"pool_{b+1}")(x)

        if spatial_dropout > 0:
            x = layers.SpatialDropout2D(
                spatial_dropout, name=f"spatial_dropout_{b+1}"
            )(x)

        filters *= 2

    x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(l2_reg) if l2_reg > 0 else None,
        name="dense",
    )(x)

    if dropout_head > 0:
        x = layers.Dropout(dropout_head, name="dropout_head")(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="model_c_random_search")
    return model


def sample_config(rng: np.random.Generator, lr_min: float, lr_max: float) -> dict:
    return {
        "n_blocks": int(rng.choice([3, 4])),
        "base_filters": int(rng.choice([32, 64])),
        "dense_units": int(rng.choice([256, 512])),
        "spatial_dropout": float(rng.choice([0.0, 0.10])),
        "dropout_head": float(rng.choice([0.3, 0.4, 0.5])),
        "l2_reg": float(rng.choice([0.0, 1e-4])),
        "learning_rate": sample_loguniform(rng, lr_min, lr_max),
        "use_batchnorm": True,
    }


def main():
    EPOCHS = 25
    EARLY_STOP_PATIENCE = 3
    N_TRIALS = 12
    SEED = 42

    LR_MIN = 1e-5
    LR_MAX = 1e-3

    train_ds, val_ds, _ = get_datasets()

    rng = np.random.default_rng(SEED)
    results = []

    for t in range(N_TRIALS):
        keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)

        cfg = sample_config(rng, LR_MIN, LR_MAX)

        aug = get_augmentation_layer()

        model = build_model_c(
            augmentation_layer=aug,
            n_blocks=cfg["n_blocks"],
            base_filters=cfg["base_filters"],
            dense_units=cfg["dense_units"],
            spatial_dropout=cfg["spatial_dropout"],
            dropout_head=cfg["dropout_head"],
            l2_reg=cfg["l2_reg"],
            use_batchnorm=cfg["use_batchnorm"],
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=float(cfg["learning_rate"])),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=0,
        )

        history = model.fit(
            train_ds,
            epochs=EPOCHS,
            validation_data=val_ds,
            callbacks=[early_stop, reduce_lr],
            verbose=0,
        )

        best_val_acc = float(np.max(history.history["val_accuracy"]))
        best_val_loss = float(np.min(history.history["val_loss"]))

        row = {
            "trial": t + 1,
            **cfg,
            "epochs": EPOCHS,
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
        }
        results.append(row)

        print(
            f"[RS] trial {t+1}/{N_TRIALS} | "
            f"val_acc={best_val_acc:.4f} | val_loss={best_val_loss:.4f} | "
            f"lr={cfg['learning_rate']:.2e} | "
            f"blocks={cfg['n_blocks']} base={cfg['base_filters']} | "
            f"sd={cfg['spatial_dropout']} dh={cfg['dropout_head']}"
        )

    results_sorted = sorted(
        results, key=lambda r: (-r["best_val_acc"], r["best_val_loss"])
    )
    best = results_sorted[0]

    with open(OUT_DIR / "rs_results_c.json", "w") as f:
        json.dump(results_sorted, f, indent=2)
    save_results_csv(OUT_DIR / "rs_results_c.csv", results_sorted)

    with open(OUT_DIR / "best_hp_c.json", "w") as f:
        json.dump(best, f, indent=2)

    print("\nbest configuration (best val_accuracy, tie-break val_loss):")
    print(json.dumps(best, indent=2))

    #retraining final model with best hyperparameters
    keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    #new augmentation instance for final model too
    aug = get_augmentation_layer()

    model = build_model_c(
        augmentation_layer=aug,
        n_blocks=int(best["n_blocks"]),
        base_filters=int(best["base_filters"]),
        dense_units=int(best["dense_units"]),
        spatial_dropout=float(best["spatial_dropout"]),
        dropout_head=float(best["dropout_head"]),
        l2_reg=float(best["l2_reg"]),
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

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    )

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    with open(OUT_DIR / "history_c.json", "w") as f:
        json.dump(history.history, f, indent=2)

    model.save(OUT_DIR / "model_c.keras")

    print("saved model, history, and random-search results in models/model_c/")


if __name__ == "__main__":
    main()
