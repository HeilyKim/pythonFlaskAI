from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
import numpy as np

# 실행마다 동일한 결과를 얻기 위해 케라스에 랜덤 시드를 사용하고
# 텐서플로 연산을 결정적으로 만듬
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
print(train_input.shape)

# _,axs = plt.subplots(1,10)
# for i in range(10):
#     axs[i].imshow(train_input[i],cmap='gray_r')
#     axs[i].axis('off')
# plt.show()

# target: 0티셔츠 1바지 2스웨터 3드레스 4코트 5샌달 6셔츠 7스니커즈 8가방 9부츠
train_X = train_input.reshape(-1,28*28)
test_X = test_input.reshape(-1,28*28)

train_scaled = train_X/255.0
test_scaled = test_X/255.0

sc = SGDClassifier(loss='log',max_iter=5,random_state=42)
sc.fit(train_scaled,train_target)

scores = cross_validate(sc,train_X,train_target,n_jobs=-1)
print('test_score:',np.mean(scores['test_score']))

pred = sc.predict((train_scaled[0]*255.0).reshape(-1,28*28))  #2차원 배열로 바꾸기
print(pred)
plt.imshow((train_scaled[0]*255.0).reshape(28,28))
plt.show()