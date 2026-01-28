from pathlib import Path
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from preprocessing.data_pipeline import get_datasets, get_augmentation_layer, IMAGE_SIZE

#project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

#output directory for model a artifacts
OUT_DIR = BASE_DIR / "models" / "model_a"

tf.keras.utils.set_random_seed(42)


NUM_CLASSES = 3

#function for model a (baseline)
def build_model_a(augmentation_layer):
    model = keras.Sequential(
        [
            #explicit input layer.
            layers.Input(
                shape=(*IMAGE_SIZE, 3),  #shape=(height, width, channels). channels=3 for rgb
                name="input_image"
            ),

            #data augmentation layer
            augmentation_layer,

            #first convolutional layer
            layers.Conv2D(
                filters=32, #keeping the model simple
                kernel_size=(3, 3), #standard choice to capture local spatial patterns such as edges and corners with few parameters
                activation="relu", #relu activation introduces non-linearity and ensures efficient training
                padding="same" #preserves spatial dimensions before pooling
            ),

            #first max-pooling layer
            layers.MaxPooling2D(
                pool_size=(2, 2) #reduces spatial resolution by a factor of 2
            ),

            #second convolutional layer
            layers.Conv2D(
                filters=64, #increased depth to learn more complex features
                kernel_size=(3, 3),
                activation="relu",
                padding="same" 
            ),

            #second max-pooling layer
            layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            #flatten layer
            layers.Flatten(),

            #dense layer
            layers.Dense(
                units=128, #moderate size
                activation="relu"
            ),

            #output layer
            layers.Dense(
                units=NUM_CLASSES,
                activation="softmax", #softmax activation (outputs class probabilities that sum to 1)
                name="pred"
            ),
        ],
      
        name="model_a_baseline",
    )

    return model

#main function
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    train_ds, val_ds, _ = get_datasets() #no test here
    aug = get_augmentation_layer()

    model = build_model_a(aug)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001), #adam is the most popular optimizer, default value for learning rate
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), #multi-class classification softmax output integer labels -> SparseCategoricalCrossentropy
        metrics=["accuracy"],
    )

    model.summary()

    #training
    history = model.fit(
    train_ds, #dataset (x, y)
    epochs=10,
    validation_data=val_ds,
    )

    #saving training metrics history for later analysis and plotting
    with open(OUT_DIR/"history_a.json", "w") as f:
        json.dump(history.history, f)

    #saving model
    model.save(OUT_DIR/"model_a.keras")

    print("Saved model and history in models/model_a/")

if __name__ == "__main__":
    main()