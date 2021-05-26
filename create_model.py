import cv2
import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, MaxPool2D, Input
from keras.applications.resnet50 import ResNet50
from keras.models import Model, load_model
import numpy as np


def create_neural_network():

    weights_path = "imagenet"
    weights_path_1 = "C:/Users/roye7/Desktop/Python Projects/Objects Relationship/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    num_classes = 4

    # optimizer - Adam, binary_crossentropy , multiy label classificatoin
    model = Sequential()
    model.add(ResNet50(include_top=False, weights=weights_path, pooling='avg'))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.layers[0].trainable = False

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def fit_model(model_, train_df_, validation_df_):
    BATCH_SIZE = 4

    image_gen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                                   zoom_range=0.2, horizontal_flip=False, vertical_flip=False, fill_mode='nearest')

    # ---- NEW ----
    train_image_gen = image_gen.flow_from_dataframe(
        train_df_,
        x_col='file_name',
        y_col='label',
        target_size=(300, 300),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_image_gen = image_gen.flow_from_dataframe(
        validation_df_,
        x_col='file_name',
        y_col='label',
        target_size=(300, 300),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    """
    # ---- OLD -----
    train_image_gen = image_gen.flow_from_directory('C:/Users/roye7/drive_images/Train', target_size=(300, 300),
                                                    batch_size=BATCH_SIZE, class_mode='categorical')

    test_image_gen = image_gen.flow_from_directory('C:/Users/roye7/drive_images/Validation', target_size=(300, 300),
                                                   batch_size=BATCH_SIZE, class_mode='categorical')
    """

    print(train_image_gen.class_indices)
    print(test_image_gen.class_indices)

    # print(model.summary())
    result = model_.fit(train_image_gen, epochs=100, validation_data=test_image_gen)

    model_.save("OurModel_NEW.h5")
    print(result)
    print(result.history['accuracy'])


def create_df():
    df_array = []

    # Go to training folder and read all inside folders
    inside_folders = ['ChairLeftTable', 'ChairRightTable', 'CupAboveChair', 'CupUnderChair']

    # Create Train dataframe
    for folder in inside_folders:
        path = f"C:\\Users\\roye7\\drive_images\\Train\\{folder}"
        files = os.listdir(path)
        df = pd.DataFrame(map(lambda s: path + "\\" + s, files), columns=["file_name"])
        df["label"] = f"[{folder}]"
        df_array.append(df)

    train_df_ = pd.concat(df_array)
    df_array.clear()

    # Create Validation dataframe
    for folder in inside_folders:
        path = f"C:\\Users\\roye7\\drive_images\\Validation\\{folder}"
        files = os.listdir(path)
        df = pd.DataFrame(map(lambda s: path + "\\" + s, files), columns=["file_name"])
        df["label"] = f"[{folder}]"
        df_array.append(df)
    validation_df_ = pd.concat(df_array)

    return train_df_, validation_df_


if __name__ == "__main__":
    """
    # Create Model:
    train_df, validation_df = create_df()

    model = create_neural_network()
    fit_model(model, train_df, validation_df)
    """

    image_gen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                                   zoom_range=0.2, horizontal_flip=False, vertical_flip=False, fill_mode='nearest')
    train_image_gen = image_gen.flow_from_directory('C:/Users/roye7/drive_images/Train', target_size=(300, 300),
                                                    batch_size=4, class_mode='categorical')
    print(train_image_gen.class_indices)

    # Use exist model:
    model = load_model('OurModel_NEW.h5')

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    img = cv2.imread('input/6.jpeg')
    img = cv2.resize(img, (300, 300))
    img = np.reshape(img, [1, 300, 300, 3])

    c = model.predict(img)
    classes = model.predict_classes(img)
    classes_proba = model.predict_proba(img)

    # get predicted class name and print:
    position = list(train_image_gen.class_indices.values()).index(classes[0])
    print(list(train_image_gen.class_indices.keys())[position])

    print(c)
    print(classes)
    print(classes_proba)
