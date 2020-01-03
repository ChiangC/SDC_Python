import csv
import cv2
import numpy as np

from keras.models import Seqential
from keras.layers import Flatten, Dese, Lambda,Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines=[]
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/'+filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

#Data augmentation        
augmented_images, augmented_measurements = [],[]
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

#Use Keras to train a network to do the following:
#1.Take in an image from the center camera of the car. This is the input to your neural network.
#2.Output a new steering angle for the car. 

model = Sequential()
# set up lambda layer
model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape=(160, 320, 3))
#Cropping2D Layer
model.add(Copping2D(cropping=((70,25),(0,0))))
# model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

#LeNet
# model.add(Convolution2D(6, 5, 5, activation="relu"))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6, 5, 5, activation="relu"))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))
          
          
          
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save("model.h5")


# with open(csv_file, 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         steering_center = float(row[3])

#         # create adjusted steering measurements for the side camera images
#         correction = 0.2 # this is a parameter to tune
#         steering_left = steering_center + correction
#         steering_right = steering_center - correction

#         # read in images from center, left and right cameras
#         path = "..." # fill in the path to your training IMG directory
#         img_center = process_image(np.asarray(Image.open(path + row[0])))
#         img_left = process_image(np.asarray(Image.open(path + row[1])))
#         img_right = process_image(np.asarray(Image.open(path + row[2])))

#         # add images and angles to data set
#         car_images.extend(img_center, img_left, img_right)
#         steering_angles.extend(steering_center, steering_left, steering_right)




























































