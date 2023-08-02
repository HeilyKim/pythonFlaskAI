import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import cv2

target = ['티셔츠', '바지', '스웨터', '드레스', '코트', '샌달', '셔츠', '스니커즈', '가방', '부츠']
model = keras.models.load_model('best_cnn.h5')
for i in range(10):
    a0 = cv2.imread('a0.png', cv2.IMREAD_GRAYSCALE)
    a0 = a0 / 255
    a0 = a0.reshape(1,28,28,1)
    pred = model.predict(a0)
    print(pred)
    print(f'{i}번째 이미지는',target[np.argmax(pred)])