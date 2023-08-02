from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, train_test_split
import numpy as np

# target: 0티셔츠 1바지 2스웨터 3드레스 4코트 5샌달 6셔츠 7스니커즈 8가방 9부츠

# 패션 데이터 로드
(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()

train_input, val_input, train_target, val_target = \
    train_test_split(train_input, train_target, test_size=0.2, random_state=42)

print(train_input.shape)
print(val_input.shape)

model = keras.Sequential([keras.layers.Dense(100, activation="sigmoid", input_shape=(784,))  # <-심층 신경망
                             , keras.layers.Dense(10, activation="softmax")])  # <-기본 신경망



# 1차원 (Flatten) == reshape안해도 됨
# model2 = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28))
#                               , keras.layers.Dense(100, activation="relu", input_shape=(784,))
#                               , keras.layers.Dense(10, activation="softmax")])
# train_scaled = train_input/ 255.0
# test_scaled = test_input/ 255.0

print(model.summary())


train_scaled = train_input.reshape(-1, 784) / 255.0
test_scaled = test_input.reshape(-1, 784) / 255.0

# optimizer: sgd(0.4649,0.8376), RMSprop(0.6204,0.7963), Adagrad(0.3811,0.8680)
model.compile(optimizer='sgd',loss="sparse_categorical_crossentropy", metrics="accuracy")
model.fit(train_scaled, train_target, epochs=5)
score = model.evaluate(test_scaled, test_target)
print(score)

# 예측 값
pred = model.predict(test_scaled[0:5])
print(pred)
# 실제 값
print(np.argmax(pred, axis=1))
print(test_target[0:5])

print(test_scaled[0].shape)  # (784,)
fig, axis = plt.subplots(1, 5)
axis[0].imshow(test_scaled[0].reshape(28, 28), cmap='gray_r')
axis[1].imshow(test_scaled[1].reshape(28, 28), cmap='gray_r')
axis[2].imshow(test_scaled[2].reshape(28, 28), cmap='gray_r')
axis[3].imshow(test_scaled[3].reshape(28, 28), cmap='gray_r')
axis[4].imshow(test_scaled[4].reshape(28, 28), cmap='gray_r')
plt.show()
