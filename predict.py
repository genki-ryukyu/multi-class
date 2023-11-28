# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 22:09:23 2020

@author: Vishal
"""

import tensorflow as tf 
classifierLoad = tf.keras.models.load_model('model1.h5')

import numpy as np
from tensorflow.keras.preprocessing import image
import glob
import cv2
from PIL import Image
import pillow_heif
from os.path import splitext


test_image = image.load_img("chukuhoutai117.jpeg", target_size = (200,200))
test_image = np.expand_dims(test_image, axis=0)
result = classifierLoad.predict(test_image)

if result[0][1] == 1:
    print("ノート型ですよ")
elif result[0][0] == 1:
    print("円柱型ですよ")
elif result[0][2] == 1:
    print("ペン型ですよ")
elif result[0][3] == 1:
    print("直方体ですよ")