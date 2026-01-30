from pathlib import Path
import tensorflow as tf


# defining configuration values
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42

# defining a fixed class order to keep label ids consistent everywhere
CLASS_NAMES = ["paper", "rock", "scissors"]

# defining project paths
BASE_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = BASE_DIR / "dataset_splits"

TRAIN_DIR = SPLITS_DIR / "train"
VAL_DIR = SPLITS_DIR / "val"
TEST_DIR = SPLITS_DIR / "test"

def get_datasets():
    # loading the training dataset from directory
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=TRAIN_DIR,
        labels="inferred", #default/could omit
        label_mode="int",  #default but better to keep
        class_names= CLASS_NAMES, #default/could omit
        color_mode="rgb", #default/could omit
        batch_size=BATCH_SIZE, #default but better to keep
        image_size=IMAGE_SIZE,
        shuffle=True, #default
        seed=SEED,
        verbose=True, #default/could omit
    )

    # loading the validation dataset from directory
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=VAL_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=False,
        verbose=True,
    )

    # loading the test dataset from directory
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=False,
        verbose=True,
    )

    # normalizing image pixel values to the range [0, 1]
    rescale = tf.keras.layers.Rescaling(scale=1./255)

    train_ds = train_ds.map(
        lambda x, y: (rescale(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_ds = val_ds.map(
        lambda x, y: (rescale(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    test_ds = test_ds.map(
        lambda x, y: (rescale(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # improving performance
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds

# defining data augmentation to be applied only during training
def get_augmentation_layer():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            ],
        name="data_augmentation",
    )

