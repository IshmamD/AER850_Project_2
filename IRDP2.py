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
