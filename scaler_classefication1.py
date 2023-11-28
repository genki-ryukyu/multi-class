                # -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 21:56:37 2020

@author: Vishal
"""

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
import os
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.optimizers import RMSprop

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 32

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        'images_scaler',  # This is the source directory for training images
        target_size=(200, 200),  # All images will be resized to 200 x 200
        batch_size=batch_size,
        # Specify the classes explicitly
        classes = ['bottle','note','pen','square'],
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')




conv_base = VGG16(weights=None,
                  include_top=False,
                  input_shape=(200,200,1))
conv_base.summary()

conv_base.trainable = False



model = Sequential()
import tensorflow as tf
model.add(conv_base)
model.add(tf.keras.layers.Convolution2D(16,3,activation='relu', input_shape=(200, 200, 3)))
model.add(tf.keras.layers.MaxPooling2D(2,2))
# model.add(tf.keras.layers.Convolution2D(32, (3,3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(2,2))
# model.add(tf.keras.layers.Convolution2D(64, (3,3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(2,2))
# model.add(tf.keras.layers.Convolution2D(64, (3,3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(2,2))
# model.add(tf.keras.layers.Convolution2D(64, (3,3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(2,2))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(4, activation='softmax'))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation="relu")) 
model.add(tf.keras.layers.Dense(64, activation="relu")) 
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(4, activation='softmax')) 


model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

model.summary()

total_sample=train_generator.n

n_epochs = 30

history = model.fit(
        train_generator, 
        steps_per_epoch=int(total_sample/batch_size),  
        epochs=n_epochs,
        verbose=1)

model.save('shin.h5')

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('3443059_LL1.jpg', target_size = (200,200))
#test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

if result[0][1] == 1:
    print("ノート型ですよ")
elif result[0][0] == 1:
    print("円柱型ですよ")
elif result[0][2] == 1:
    print("ペン型ですよ")
elif result[0][3] == 1:
    print("直方体ですよ")



