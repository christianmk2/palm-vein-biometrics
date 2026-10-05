"""
Trains a VGG16-based transfer-learning classifier on the preprocessed palm
image dataset (see preprocess_dataset.py) and reports the usual
classification metrics (precision/recall/F1, confusion matrix, FAR/FRR/EER).
"""

import os

import numpy as np
import cv2
from PIL import Image

import matplotlib.pyplot as plt
plt.style.use("dark_background")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score,
)

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# Previous dataset locations used while iterating on preprocessing during
# the project (Pasig/Quezon batches, different user counts and image
# counts per class) - kept here for reference:
#   D:/PALM QUEZON/PALM QUEZON DATASET/Pasig_April 27/Preprocessed Palm Data/...
#   D:/PALM QUEZON/PALM QUEZON DATASET/Pasig_May 01/Preprocessed/5 Users (200)/
#   D:/PALM QUEZON/PALM QUEZON DATASET/Pasig_May 02 Testing/Preprocessed/...
DATASET_DIR = "D:/PALM QUEZON/PALM QUEZON DATASET/May 5/preprocess/"
MODEL_OUT_PATH = "D:/PALM QUEZON/PALM QUEZON DATASET/Pasig_Last/h5 last/3last.h5"
CHECKPOINT_PATH = "vgg16_finetune.h5"

RANDOM_SEED = 42
IMAGE_SIZE = (224, 224)
TEST_SIZE = 0.2
EPOCHS = 7
BATCH_SIZE = 32
LEARNING_RATE = 0.00005
FROZEN_VGG_LAYERS = 15  # first N VGG16 layers kept frozen


def load_images_and_labels(dataset_dir, categories):
    images, labels = [], []
    for label_index, category in enumerate(categories):
        category_dir = os.path.join(dataset_dir, category)
        for image_name in os.listdir(category_dir):
            img = cv2.imread(os.path.join(category_dir, image_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = Image.fromarray(img, "RGB").resize(IMAGE_SIZE)
            images.append(np.array(resized))
            labels.append(label_index)
    return images, labels


def build_model(num_categories):
    vgg = VGG16(input_shape=IMAGE_SIZE + (3,), weights="imagenet", include_top=False)
    for layer in vgg.layers[:FROZEN_VGG_LAYERS]:
        layer.trainable = False

    x = Flatten()(vgg.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x)
    prediction = Dense(num_categories, activation="softmax")(x)

    model = Model(inputs=vgg.input, outputs=prediction)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.legend()
    plt.savefig("LossVal_loss")  # save before show() - some backends clear the figure on show()
    plt.show()

    plt.plot(history.history["accuracy"], label="train acc")
    plt.plot(history.history["val_accuracy"], label="val acc")
    plt.legend()
    plt.savefig("AccVal_acc")
    plt.show()


def print_confusion_metrics(y_true, y_pred, categories):
    print(classification_report(y_true, y_pred, target_names=categories))
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    cm_display = confusion_matrix(y_pred, y_true)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_display, display_labels=categories)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)

    # Per-class confusion-matrix-derived counts
    fp = (cm.sum(axis=0) - np.diag(cm)).astype(int)
    fn = (cm.sum(axis=1) - np.diag(cm)).astype(int)
    tp = np.diag(cm).astype(int)
    tn = (cm.sum() - (fp + fn + tp)).astype(int)

    print("Worst-class index by FP:", np.argmax(fp))
    print("Worst-class index by FN:", np.argmax(fn))
    print("Best-class index by TP:", np.argmax(tp))
    print("Best-class index by TN:", np.argmax(tn))
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)
    print("TN:", tn)

    # Confusion-matrix-based rates
    fpr_per_class = fp / (fp + tn)
    fnr_per_class = fn / (tp + fn)
    print("FPR per class:", fpr_per_class)
    print("FNR per class:", fnr_per_class)

    far = fpr_per_class.mean()
    frr = fnr_per_class.mean()
    eer = (far + frr) / 2
    print("FAR:", far)
    print("FRR:", frr)
    print("EER:", eer)

    # sklearn-based macro precision/recall/F1 (cross-check against the above)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    print("Classification report (sklearn macro averages)")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("FPR (1 - recall):", 1 - recall)
    print("FNR (1 - recall, true-positive side):", 1 - recall)


def main():
    categories = os.listdir(DATASET_DIR)
    print("List of categories =", categories, "\n\nNo. of categories =", len(categories))

    images, labels = load_images_and_labels(DATASET_DIR, categories)
    print("No. of images loaded =", len(images), "\nNo. of labels loaded =", len(labels))

    images = np.array(images)
    labels = np.array(labels)
    print("Images shape =", images.shape, "\nLabels shape =", labels.shape)

    np.random.seed(RANDOM_SEED)
    shuffle_order = np.arange(images.shape[0])
    np.random.shuffle(shuffle_order)
    images = images[shuffle_order]
    labels = labels[shuffle_order]

    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train_categorical = to_categorical(y_train)
    y_test_categorical = to_categorical(y_test)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255, rotation_range=30,
        width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.1, brightness_range=[0.7, 1.3],
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow(x_train, y_train_categorical, batch_size=BATCH_SIZE)
    test_set = test_datagen.flow(x_test, y_test_categorical, batch_size=BATCH_SIZE)

    model = build_model(len(categories))
    model.summary()

    # Callbacks are defined but left disabled below, same as the original -
    # note ModelCheckpoint's mode should be "min" (not "max") if you enable
    # it, since val_loss should be minimized.
    checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor="val_loss", mode="min", save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=0.00001, mode="min")

    history = model.fit(
        training_set,
        validation_data=test_set,
        epochs=EPOCHS,
        steps_per_epoch=len(training_set),
        validation_steps=len(test_set),
        # callbacks=[reduce_lr, checkpoint],
        shuffle=True,
    )

    plot_history(history)

    test_loss, test_accuracy = model.evaluate(test_set)
    print("Test:", test_loss, test_accuracy)
    train_loss, train_accuracy = model.evaluate(training_set)
    print("Train:", train_loss, train_accuracy)

    model.save(MODEL_OUT_PATH)
    reloaded_model = keras.models.load_model(MODEL_OUT_PATH)

    y_predict = reloaded_model.predict(x_test).argmax(axis=1)
    print_confusion_metrics(y_test, y_predict, categories)


if __name__ == "__main__":
    main()
