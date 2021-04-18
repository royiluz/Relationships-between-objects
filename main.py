import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, MaxPool2D, Input
from keras.applications.resnet50 import ResNet50
from keras.models import Model, load_model
import numpy as np


def create_neural_network():

    weights_path = "imagenet"
    weights_path_1 = "C:/Users/roye7/Desktop/Python Projects/Objects Relationship/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    num_classes = 2

    model = Sequential()
    model.add(ResNet50(include_top=False, weights=weights_path, pooling='avg'))
    model.add(Dense(num_classes, activation='softmax'))
    model.layers[0].trainable = False

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def fit_model(model_):
    BATCH_SIZE = 4

    image_gen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                                   zoom_range=0.2, horizontal_flip=True, vertical_flip=False, fill_mode='nearest')

    train_image_gen = image_gen.flow_from_directory('C:/Users/roye7/drive_images/Train', target_size=(300, 300),
                                                    batch_size=BATCH_SIZE, class_mode='categorical')

    test_image_gen = image_gen.flow_from_directory('C:/Users/roye7/drive_images/Validation', target_size=(300, 300),
                                                   batch_size=BATCH_SIZE, class_mode='categorical')

    print(train_image_gen.class_indices)
    print(test_image_gen.class_indices)

    # print(model.summary())
    result = model_.fit(train_image_gen, epochs=100, validation_data=test_image_gen)

    # model.save("OurModel.h5")
    print(result)
    print(result.history['accuracy'])


if __name__ == "__main__":
    # model = create_neural_network()

    image_gen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2,
                                   zoom_range=0.2, horizontal_flip=True, vertical_flip=False, fill_mode='nearest')
    train_image_gen = image_gen.flow_from_directory('C:/Users/roye7/drive_images/Train', target_size=(300, 300),
                                                    batch_size=4, class_mode='categorical')
    print(train_image_gen.class_indices)

    model = load_model('models/CupUnderAboveChair.h5')

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    img = cv2.imread('2_above.jpeg')
    img = cv2.resize(img, (300, 300))
    img = np.reshape(img, [1, 300, 300, 3])

    classes = model.predict_classes(img)
    classes_proba = model.predict_proba(img)

    # get predicted class name and print:
    position = list(train_image_gen.class_indices.values()).index(classes[0])
    print(list(train_image_gen.class_indices.keys())[position])

    # print(classes)
    print(classes_proba)
