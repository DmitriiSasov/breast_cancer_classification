import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from keras.optimizer_v1 import Adam
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Sequential


data_dir = fr'F:\Dima\dissertation\Data\other_datasets\some_paper\all_data'
img_height = 2048
img_width = 1536
batch_size = 32

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(6):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(train_ds.class_names[labels[i]])
            plt.axis("off")

    resnet_model = Sequential()

    pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                      input_shape=(img_height, img_width, 3),
                                                      pooling='avg', classes=4,
                                                      weights='imagenet')
    for layer in pretrained_model.layers:
        layer.trainable = False

    resnet_model.add(pretrained_model)
    resnet_model.add(Flatten())
    resnet_model.add(Dense(512, activation='relu'))
    resnet_model.add(Dense(4, activation='softmax'))

    resnet_model.summary()

    resnet_model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    history = resnet_model.fit(train_ds, validation_data=val_ds, epochs=10)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
