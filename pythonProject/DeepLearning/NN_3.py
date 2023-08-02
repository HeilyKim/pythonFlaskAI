import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

#모델을 설계하는 함수
def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    model.add(keras.layers.Dense(100,activation="relu"))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10,activation="softmax"))
    return model
model =  model_fn(keras.layers.Dropout(0.3))
print(model.summary())

model.compile(
    optimizer="adam"
    ,loss="sparse_categorical_crossentropy"
    ,metrics="accuracy"
)
#가장 좋은 모델 저장
checkPoint = keras.callbacks.ModelCheckpoint('best_model.h5',save_best_only=True)
earlyStopping = keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)

# epochs를 무한정으로 하면 무한정 학습이됨(x) #verbose = 0 학습 진행 상태 안 보여줌(보고싶으면 =1  => 어느 지짐이 괜찮은지도 볼수있음)
history = model.fit(train_scaled,train_target
                    ,epochs=20,verbose=0
                    ,validation_data=[val_scaled,val_target]
                    ,callbacks=[checkPoint,earlyStopping])
print(history.history.keys())
"""
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show() #loss 가 작을수로 정확도에 가까움 # 그래프로 어느 지점이 최적인지 보여줌

plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
"""

model.save_weights("model_weights.h5") #모델 가중치
model.save('model_whole.h5')



