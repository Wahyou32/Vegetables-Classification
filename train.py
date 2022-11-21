import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import copy, cv2, glob, shutil
import os
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import zipfile
from sklearn.model_selection import train_test_split


class CB(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.90):
            self.model.stop_training = True

STOP = CB()


zipped = 'vegetables.zip'
reference = zipfile.ZipFile(zipped, 'r')
reference.extractall('/tmp')
reference.close()


directory = '/tmp/Vegetable Images'
train_dir = os.path.join(directory, 'train')
val_dir = os.path.join(directory, 'validation')

train_repo_bean = os.path.join(train_dir, 'bean')
train_repo_bitter_gourd = os.path.join(train_dir, 'bitter_gourd')
train_repo_bottle_gourd = os.path.join(train_dir, 'bottle_gourd')
train_repo_brinjal = os.path.join(train_dir, 'brinjal')
train_repo_broccoli = os.path.join(train_dir, 'broccoli')
train_repo_cabbage = os.path.join(train_dir, 'cabbage')
train_repo_capsicum = os.path.join(train_dir, 'capsicum')
train_repo_carrot = os.path.join(train_dir, 'carrot')
train_repo_cauliflower = os.path.join(train_dir, 'cauliflower')
train_repo_cucumber = os.path.join(train_dir, 'cucumber')
train_repo_papaya = os.path.join(train_dir, 'papaya')
train_repo_potato = os.path.join(train_dir, 'potato')
train_repo_pumpkin = os.path.join(train_dir, 'pumpkin')
train_repo_radish = os.path.join(train_dir, 'radish')
train_repo_tomato = os.path.join(train_dir, 'tomato')


train_bean = train_test_split(train_repo_bean, test_size=0.40)
train_bitter_gourd = train_test_split(train_repo_bitter_gourd, test_size=0.40)
train_bottle_gourd = train_test_split(train_repo_bottle_gourd, test_size=0.40)
train_brinjal = train_test_split(train_repo_brinjal, test_size=0.40)
train_broccoli = train_test_split(train_repo_broccoli, test_size=0.40)
train_cabbage = train_test_split(train_repo_cabbage, test_size=0.40)
train_capsicum = train_test_split(train_repo_capsicum, test_size=0.40)
train_carrot = train_test_split(train_repo_carrot, test_size=0.40)
train_cauliflower = train_test_split(train_repo_cauliflower, test_size=0.40)
train_cucumber = train_test_split(train_repo_cucumber, test_size=0.40)
train_papaya = train_test_split(train_repo_papaya, test_size=0.40)
train_potato = train_test_split(train_repo_potato, test_size=0.40)
train_pumpkin = train_test_split(train_repo_pumpkin, test_size=0.40)
train_radish = train_test_split(train_repo_radish, test_size=0.40)
train_tomato = train_test_split(train_repo_tomato, test_size=0.40)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range=0.35,
    fill_mode='nearest'
)


val_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.35,
    fill_mode='nearest'
)


train_model = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

val_model = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(15, activation='softmax')
])


model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(),
    metrics=['accuracy']
)

trained_model = model.fit(
    train_model,
    steps_per_epoch=35,
    epochs=20,
    validation_data=val_model,
    verbose=2,
    callbacks=[STOP]
)


export_path = 'vegetable_model'

tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)
