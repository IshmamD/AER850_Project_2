# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 18:51:18 2024

@author: ishma
"""

#STEP 5

from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the saved model
model = load_model('./trained_model.h5')


test_dir = './P2D/Data/test/missing-head/IMG_20230511_100229_jpg.rf.08e4a8127f1d2057801e4a7087862f85.jpg'

img = image.load_img(test_dir,target_size=(500,500))
x = image.img_to_array(img)
x = x/255
x = np.expand_dims(x, axis=0)
classes = ['crack', 'missing-head', 'paint-off']
prediction = model.predict(x)
print(prediction)
predicted_index = np.argmax(prediction[0])
print(predicted_index)
predicted_class = classes[predicted_index]
print(predicted_class)

plt.figure()
plt.imshow(img)

prediction = 'Crack: 0.05% \nMissing-Head: 99.93% \nPaint-off: 0.02%' # I ran the code once to see the values and
#manually converted the scientific notation to more readable percentages

plt.text(60,100,prediction,color='green')
plt.axis('off')

model2 = load_model('./trained_model2.h5')
prediction2 = model2.predict(x)
print(prediction2)
predicted_index2 = np.argmax(prediction2[0])
print(predicted_index2)
predicted_class2 = classes[predicted_index2]
print(predicted_class2)


plt.figure()
plt.imshow(img)
plt.axis('off')

prediction2 =  'Crack: 3.04% \nMissing-Head: 95.63% \nPaint-off: 1.33%'
plt.text(60,100,prediction2,color='green')
