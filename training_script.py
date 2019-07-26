# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:00:57 2019

@author: Olusegun Folarin
"""

# Convolutional Neural Network

# Part I- building CNN
# Import the necessary libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
# assign variable input_shape to be used in later stages
input_shape = (180, 180)

classifier = Sequential()
# adding layers
# step 1- Convolution
classifier.add(Conv2D(filters=32,  #number of filters/feature maps to use
                             kernel_size=(3, 3),  #size of the filter
                             input_shape=(*input_shape, 3),  #shape of the images
                             activation='relu'))  # activation
# step 2- Max Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))  # Average Pooling possible as well


#second convolution with 32 filters and a kernel size of 3,3
classifier.add(Conv2D(filters= 32, #number of filters/feature maps to use
                             kernel_size= (3, 3), #size of the filter
                             activation='relu'))#activation
#second max-pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))


#third convolution with 32 filters
classifier.add(Conv2D(filters= 32, #number of filters/feature maps to use
                             kernel_size= (3,3), #size of the filter
                             activation= 'relu'))#activation
#third max pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Fourth convolution with 64 filters.
classifier.add(Conv2D(filters=64,  # number of filters/feature maps to use
                             kernel_size=(3,3),  # size of the filter
                             activation='relu'))  #activation
# Fourth max pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))


# step 3- Flattening
classifier.add(Flatten())

# step 4- Making the Full Connection with three 
# Dense layers and two dropout for regularization and overfitting reduction
classifier.add(Dense(64,
                     activation='relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(64,
                     activation='relu'))
classifier.add(Dense(64,
                     activation='relu'))
classifier.add(Dropout(0.1))
classifier.add(Dense(1,
                     activation='sigmoid'))
# compiling
optimizer = Adam(lr=1e-3)
classifier.compile(optimizer=optimizer, loss='binary_crossentropy',
                   metrics=['accuracy'])  # Binary crosentropy
# since it is two categories

# Using Keras Image Augmentation to further increase the
# training set size, by introducing augmentation and flipping

train_datagen = ImageDataGenerator(
                                rescale=1./255,  # important to scale
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# image flow from directory
training_set = train_datagen.flow_from_directory(
                                                'dataset/training_set',
                                                target_size=input_shape,
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                            'dataset/test_set',
                                            target_size=input_shape,
                                            batch_size=32,
                                            class_mode='binary')

# fit classifier
classifier.fit_generator(
                        training_set,
                        steps_per_epoch=8000/32,
                        epochs=55,
                        validation_data=test_set,
                        validation_steps=2000/32)

# Save model in hdf5 format
classifier.save('cnn.h5')

