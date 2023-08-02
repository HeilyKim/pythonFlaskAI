import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import cv2
# target: 0티셔츠 1바지 2스웨터 3드레스 4코트 5샌달 6셔츠 7스니커즈 8가방 9부츠
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()
#0 ~ 255
train_scaled = train_input.reshape(-1, 28, 28, 1)

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

for i in range(10):
    cv2.imwrite(f'a{i}.png',val_scaled[i].reshape(28,28))



# fig,axs = plt.subplots(1,10)
# for i in range(10):
#     axs[i].imshow(val_scaled[i].reshape(28,28),cmap='gray_r')
# plt.show()
# for i in range(10):
#     plt.imshow(val_scaled[i].reshape(28,28))
#     plt.savefig(f'a{i}.png')


# model = keras.models.Sequential()
# model.add(keras.layers.Conv2D(32,kernel_size=(3,3)
#                               ,input_shape=(28,28,1)
#                               ,activation="relu"
#                               ,padding="same"))
# model.add(keras.layers.MaxPool2D(2))
#
# model = keras.models.Sequential()
# model.add(keras.layers.Conv2D(64,kernel_size=(3,3)
#                               ,input_shape=(28,28,1)
#                               ,activation="relu"
#                               ,padding="same"))
# model.add(keras.layers.MaxPool2D(2))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(100,activation="relu"))
# model.add(keras.layers.Dropout(0.3))
# model.add(keras.layers.Dense(10,activation="softmax"))
# print(model.summary())
#
# model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics="accuracy")
# checkpoint = keras.callbacks.ModelCheckpoint('best_cnn.h5',save_best_only=True)
# earlyStopping = keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)
# model.fit(train_scaled,train_target
#           ,epochs=20,validation_data=(val_scaled,val_target)
#           ,callbacks=[checkpoint,earlyStopping])