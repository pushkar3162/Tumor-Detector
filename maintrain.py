import cv2
import os

import numpy as np
from PIL import Image
import tensorflow as tf
from keras.utils import normalize
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D , MaxPool2D
from keras.layers import Activation, Dropout , Flatten, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dropout


image_directory = 'dataset/'
no_tumour_images = os.listdir(os.path.join(image_directory, 'no/'))
yes_tumour_images = os.listdir(os.path.join(image_directory, 'yes/'))

# min_images = min(len(no_tumour_images), len(yes_tumour_images))
# no_tumour_images = no_tumour_images[:min_images]
# yes_tumour_images = yes_tumour_images[:min_images]

dataset = []
label = []

INPUT_SIZE = 64
#print(no_tumour_images)
#path = 'no0.jpg'
#print(path.split('.')[1])

for i , image_name in enumerate(no_tumour_images):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory+'no/'+image_name)
        image = Image.fromarray(image,'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i , image_name in enumerate(yes_tumour_images):
    if(image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory+'yes/'+image_name)
        image = Image.fromarray(image,'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

#print(len(dataset))
#print(len(label))

dataset = np.array(dataset)
label = np.array(label)
#dataset, label = shuffle(dataset, label, random_state=0)
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)
#Reshape = (n,image_width, image_height, n_channel)
#print(x_train.shape)
#print(y_test.shape)
# Reshape input data for normalization
# Normalize input data
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

#y_train = to_categorical(y_train , num_classes=2)
#y_test = to_categorical(y_test , num_classes=2)


# Model Building
# 64,64,3
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(INPUT_SIZE,INPUT_SIZE,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Binary cross entropy
# Binary CrossEntropy = 1 ,sigmoid
# Catagorical Cross Entryopy = 2, softmax

model.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics = ['accuracy'])

model.fit(x_train, y_train , batch_size = 16, verbose = 1, epochs = 10, validation_data =(x_test,y_test), shuffle = False)
model.save('BrainTumour10Epochs.h5')