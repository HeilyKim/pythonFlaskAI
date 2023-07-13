from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
data = [[l,w] for l,w in zip(fish_length,fish_weight)]
target = ["도미"]*35+["빙어"]*14
kn = KNeighborsClassifier()
##==random하지 않았을때 (score=0)==##
# trainData = data[:35]
# trainTarget = target[:35]
# testData = data[35:]
# testTarget = target[35:]
# kn.fit(trainData,trainTarget)
# score = kn.score(testData,testTarget)
# # print(score)
# pred = kn.predict([[5,30]])
# # print(f'예측값: {pred}')

inputArr = np.array(data)
targetArr = np.array(target)
# print(inputArr.shape)#(49,2)2개의 데이터가 하나로 이루어짐 총 49개
# print(targetArr.shape)
np.random.seed(42) #랜덤패턴42번이 나옴 (모두가 다 같음) (이걸 안하면 매번 학습데이터가 다르기에 정확도가 항상 틀려짐)
index = np.arange(49) #0-48까지 나옴
np.random.shuffle(index)

trainArr = inputArr[index[:35]]
trainArrT = targetArr[index[:35]]

testArr = inputArr[index[35:]]
testArrT = targetArr[index[35:]]

plt.scatter(trainArr[:,0],trainArr[:,1]) #trainArr의 x,y
plt.scatter(testArr[:,0],testArr[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
kn.fit(trainArr,trainArrT)
score = kn.score(testArr,testArrT)
print(score)