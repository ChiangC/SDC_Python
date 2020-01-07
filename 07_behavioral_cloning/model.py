import csv
import cv2
import numpy as np
import os
from math import ceil 
from PIL import Image
import random

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
       
########################################################################
#Model Architecture
########################################################################
#Use Keras to train a network to do the following:
#1.Take in an image from the center camera of the car. This is the input to your neural network.
#2.Output a new steering angle for the car. 

model = Sequential()
#Set up lambda layer
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
#Cropping2D Layer
# model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(48, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

########################################################################
#Data augmentation        
########################################################################
def data_augment(images, measurements):
    augmented_images, augmented_measurements = [],[]
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement*-1.0)

    return augmented_images,augmented_measurements

########################################################################
#Generator
########################################################################
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for row in batch_samples:
                steering_center = float(row[3])

                # create adjusted steering measurements for the side camera images
                correction = 0.1 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                
#                 steering_left = steering_center
#                 steering_right = steering_center
                # read in images from center, left and right cameras
                path = "./data/IMG/" 
                center_path = path + row[0].split(os.sep)[-1]
                left_path = path + row[1].split(os.sep)[-1]
                right_path = path + row[2].split(os.sep)[-1]
                if os.path.exists(center_path) and os.path.exists(left_path) and os.path.exists(right_path):
                    img_center = np.asarray(Image.open(center_path))
                    img_left = np.asarray(Image.open(left_path))
                    img_right = np.asarray(Image.open(right_path))

                    # add images and angles to data set
                    images.append(img_center)
                    images.append(img_left)
                    images.append(img_right)
                    angles.append(steering_center)
                    angles.append(steering_left)
                    angles.append(steering_right)

            augmented_images, augmented_measurements = data_augment(images, angles)
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)
            
            
########################################################################
########################################################################
samples=[]
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
train_samples, validation_samples = train_test_split(samples, test_size=0.3)

batch_size = 32

#Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)          
          
model.summary()   

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)
history_object = model.fit_generator(train_generator, 
                                     steps_per_epoch=ceil(len(train_samples)/batch_size),
                                     validation_data=validation_generator, 
                                     validation_steps=ceil(len(validation_samples)/batch_size), 
                                     epochs=5, verbose=1)



model.save("model.h5")




          
          