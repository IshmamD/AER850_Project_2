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


test_dir = './P2D/Data/test'

