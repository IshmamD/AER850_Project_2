# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 18:51:18 2024

@author: ishma
"""

#STEP 5

from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image


# Load the saved model
model = load_model('./trained_model.h5')


test_dir = './P2D/Data/test/missing-head/IMG_20230511_100229_jpg.rf.08e4a8127f1d2057801e4a7087862f85.jpg'

img = image.load_img(test_dir,target_size=(500,500))
x = image.img_to_array(img)
