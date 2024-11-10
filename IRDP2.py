# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 23:20:00 2024
AER850 Project 2
@author: Ishmam Raza Dewan
500956014
Section 2
"""

import numpy as np
import pandas as pd
from keras import layers
from keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#STEP 1

input_shape = (500,500,3)

train_dir = './P2D/Data/train'
val_dir = './P2D/Data/valid'
test_dir = './P2D/Data/test'

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical')


val_datagen = ImageDataGenerator(rescale = 1./255)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical')

#STEP 2

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout())
model.add(layers.Dense(3, activation='softmax'))

model.summary()

