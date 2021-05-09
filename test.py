import cv2 as cv
import tensorflow as tf
import numpy as np
import pandas
from PIL import Image
import matplotlib.pyplot as plt

classes = {0: 'Speed limit (20km/h)',
           1: 'Speed limit (30km/h)',
           2: 'Speed limit (50km/h)',
           3: 'Speed limit (60km/h)',
           4: 'Speed limit (70km/h)',
           5: 'Speed limit (80km/h)',
           6: 'End of speed limit (80km/h)',
           7: 'Speed limit (100km/h)',
           8: 'Speed limit (120km/h)',
           9: 'No passing',
           10: 'No passing veh over 3.5 tons',
           11: 'Right-of-way at intersection',
           12: 'Priority road',
           13: 'Yield',
           14: 'Stop',
           15: 'No vehicles',
           16: 'Veh > 3.5 tons prohibited',
           17: 'No entry',
           18: 'General caution',
           19: 'Dangerous curve left',
           20: 'Dangerous curve right',
           21: 'Double curve',
           22: 'Bumpy road',
           23: 'Slippery road',
           24: 'Road narrows on the right',
           25: 'Road work',
           26: 'Traffic signals',
           27: 'Pedestrians',
           28: 'Children crossing',
           29: 'Bicycles crossing',
           30: 'Beware of ice/snow',
           31: 'Wild animals crossing',
           32: 'End speed + passing limits',
           33: 'Turn right ahead',
           34: 'Turn left ahead',
           35: 'Ahead only',
           36: 'Go straight or right',
           37: 'Go straight or left',
           38: 'Keep right',
           39: 'Keep left',
           40: 'Roundabout mandatory',
           41: 'End of no passing',
           42: 'End no passing veh > 3.5 tons'}


CNN = tf.keras.models.load_model('mymodel')


test = pandas.read_csv('E:/~PI/Traffic_Sign_Detection/german/Test.csv')
labels = test["ClassId"].values
test_imgs = test["Path"].values

img_index = 111
image = Image.open('E:/~PI/Traffic_Sign_Detection/german/' + test_imgs[img_index])
img = image.resize((32, 32))
img = np.array(img) / 255.
img = img.reshape(1, 32, 32, 3)
print(img.shape)
print(labels[img_index])
print(int(np.argmax(CNN.predict(img), axis=-1)))
print(classes[int(np.argmax(CNN.predict(img), axis=-1))])
print()
plt.imshow(image)
plt.show()



image = Image.open('E:/~PI/Traffic_Sign_Detection/asd5.bmp')
img = image.resize((32, 32))
img = np.array(img) / 255.
img = img.reshape(1, 32, 32, 3)
print(img.shape)
print(int(np.argmax(CNN.predict(img), axis=-1)))
print(classes[int(np.argmax(CNN.predict(img), axis=-1))])
print()
plt.imshow(image)
plt.show()