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


test_dir = './P2D/Data/test/missing-head/test_missinghead.jpg'

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

prediction = 'Crack: 17.502646% \nMissing-Head: 80.81511% \nPaint-off: 1.682236%' # I ran the code once to see the values and
#manually converted the scientific notation to more readable percentages

plt.text(60,100,prediction,color='white')
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

prediction2 =  'Crack: 9.33422% \nMissing-Head: 85.48511% \nPaint-off: 5.180668%'
plt.text(60,100,prediction2,color='white')




test_dir = './P2D/Data/test/crack/test_crack.jpg'




img = image.load_img(test_dir,target_size=(500,500))
x = image.img_to_array(img)
x = x/255
x = np.expand_dims(x, axis=0)

prediction = model.predict(x)
print(prediction)
predicted_index = np.argmax(prediction[0])
print(predicted_index)
predicted_class = classes[predicted_index]
print(predicted_class)




plt.figure()
plt.imshow(img)
prediction = 'Crack: 53.78876% \nMissing-Head: 0.596226% \nPaint-off: 45.615023%' 

plt.text(60,100,prediction,color='green')
plt.axis('off')

prediction2 = model2.predict(x)
print(prediction2)
predicted_index2 = np.argmax(prediction2[0])
print(predicted_index2)
predicted_class2 = classes[predicted_index2]
print(predicted_class2)


plt.figure()
plt.imshow(img)
plt.axis('off')

prediction2 =  'Crack: 41.18915% \nMissing-Head: 0.415047% \nPaint-off: 58.3958%'
plt.text(60,100,prediction2,color='green')
#PAINT OFF

test_dir = './P2D/Data/test/paint-off/test_paintoff.jpg'
img = image.load_img(test_dir,target_size=(500,500))
x = image.img_to_array(img)
x = x/255
x = np.expand_dims(x, axis=0)

prediction = model.predict(x)
print(prediction)
predicted_index = np.argmax(prediction[0])
print(predicted_index)
predicted_class = classes[predicted_index]
print(predicted_class)




plt.figure()
plt.imshow(img)
prediction = 'Crack: 58.11675% \nMissing-Head: 4.769205% \nPaint-off: 37.11404%' 

plt.text(60,100,prediction,color='green')
plt.axis('off')

prediction2 = model2.predict(x)
print(prediction2)
predicted_index2 = np.argmax(prediction2[0])
print(predicted_index2)
predicted_class2 = classes[predicted_index2]
print(predicted_class2)


plt.figure()
plt.imshow(img)
plt.axis('off')

prediction2 =  'Crack: 45.050034% \nMissing-Head: 11.775294% \nPaint-off: 43.174672%'
plt.text(60,100,prediction2,color='green')
