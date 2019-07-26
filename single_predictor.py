# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:56:39 2019

@author: Olusegun Folarin
"""
# single prediction
from keras.preprocessing import image
from keras.models import load_model
import numpy as np

# Load model
classifier = load_model('cnn.h5')
def single_predictor(dir='dataset/single_prediction/cat_or_dog_1.jpg'):
    test_image = image.load_img(dir, target_size=(180, 180))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    print(f'The image is a {prediction} image.')
    return prediction


if __name__ == '__main__':
    single_predictor()
