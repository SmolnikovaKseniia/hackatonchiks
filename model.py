import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from tensorflow.keras.layers import Input, Conv2D,Concatenate, MaxPooling2D,Conv2DTranspose, MaxPool2D, Dropout, UpSampling2D, concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Model

global image_h
global image_w
global num_classes
global classes
global rgb_codes

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path):
    train_x = sorted(glob(os.path.join(path, "test", "images", "*.jpg")))
    train_y = sorted(glob(os.path.join(path, "test", "labels", "*.png")))

    valid_x = sorted(glob(os.path.join(path, "val", "images", "*.jpg")))
    valid_y = sorted(glob(os.path.join(path, "val", "labels", "*.png")))

    test_x = sorted(glob(os.path.join(path, "train", "images", "*.jpg")))
    test_y = sorted(glob(os.path.join(path, "train", "labels", "*.png")))

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape, num_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs)
    return model

def preprocess(x_path, y_path):
    # Read image and mask from file paths
    x = tf.io.read_file(x_path)
    x = tf.image.decode_jpeg(x, channels=3)  # Assuming RGB image
    x = tf.image.rgb_to_grayscale(x)  # Convert RGB to grayscale
    x = tf.image.resize(x, [image_h, image_w])
    x = tf.cast(x, tf.float32) / 255.0

    y = tf.io.read_file(y_path)
    y = tf.image.decode_png(y, channels=1)  # Assuming grayscale mask
    y = tf.image.resize(y, [image_h, image_w])
    y = tf.cast(y, tf.int32)

    # One-hot encode the mask tensor
    y = tf.one_hot(y, depth=num_classes)

    # Remove the extra dimension from the target tensor
    y = tf.squeeze(y, axis=-2)

    return x, y

def tf_dataset(X, Y, batch=8):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.shuffle(buffer_size=5000)
    ds = ds.map(preprocess)
    ds = ds.batch(batch).prefetch(2)
    return ds

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    image_h = 256
    image_w = 256
    num_classes = 11
    input_shape = (image_h, image_w, 1)
    batch_size = 8
    lr = 1e-4
    num_epochs = 25

    """ Paths """
    dataset_path = "/kaggle/input/dataseet/LaPa"
    model_path = os.path.join("files", "model.keras")
    csv_path = os.path.join("files", "data.csv")

    """ RGB Code and Classes """
    rgb_codes = [
        [0, 0, 0], [0, 153, 255], [102, 255, 153], [0, 204, 153],
        [255, 255, 102], [255, 255, 204], [255, 153, 0], [255, 102, 255],
        [102, 0, 51], [255, 204, 255], [255, 0, 102]
    ]

    classes = [
        "background", "skin", "left eyebrow", "right eyebrow",
        "left eye", "right eye", "nose", "upper lip", "inner mouth",
        "lower lip", "hair"
    ]

    """ Loading the dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")
    print("")

    """ Dataset Pipeline """
    train_ds = tf_dataset(train_x, train_y, batch=batch_size)
    valid_ds = tf_dataset(valid_x, valid_y, batch=batch_size)

    """ Model """
    model = build_unet(input_shape, num_classes)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr)
    )

    """ Training """
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path, append=True),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=num_epochs,
        callbacks=callbacks
    )