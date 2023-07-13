import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()
# print(fish.head())
# print(pd.unique(fish['Species']))
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
# print(fish_input[:5])
fish_target = fish['Species'].to_numpy()
print(fish_target)

trainX, testX, trainY, testY = train_test_split(fish_input, fish_target, random_state=42)
ss = StandardScaler()
ss.fit(trainX)
trainScaled = ss.transform(trainX)
testScaled = ss.transform(testX)

print(trainScaled)
print(testScaled)

# kn = KNeighborsClassifier(n_neighbors=3)
# kn.fit(trainScaled, trainY)
# trainScore = kn.score(trainScaled, trainY)
# testScore = kn.score(testScaled, testY)
# print(trainScore)
# print(testScore)

z = np.arange(-5, 5, 0.1)  # -5~5까지 0.1식 증가
# print(np.round(z,decimals=2))
phi = 1 / (1 + np.exp(-z))  # sigmoid 함수, 숫자를 0~1사이로 바꿈
# print(np.round(phi,decimals=2))
# plt.plot(z, phi)
# plt.xlabel('z')
# plt.ylabel('phi')
# plt.show()

# charArr = np.array(['A', 'B', 'C', 'D', 'E'])
# print(charArr[[True,False,True,False,False]]) ##True인 A,C만 출력됨

##도미 빙어만 분류해보자아
breamSmeltIndex = (trainY == 'Bream') | (trainY == 'Smelt')
print(breamSmeltIndex)
trainBreamSmelt = trainScaled[breamSmeltIndex]
targetBreamSmelt = trainY[breamSmeltIndex]
# print(trainBreamSmelt[:5])
# print(targetBreamSmelt[:5])
lr = LogisticRegression()
lr.fit(trainBreamSmelt,targetBreamSmelt)
score = lr.score(trainBreamSmelt,targetBreamSmelt)
print(score)
lr = LogisticRegression()
lr.fit(trainScaled, trainY)
trainPred = lr.score(trainScaled, trainY)
testPred = lr.score(testScaled, testY)
print(trainPred)
print(testPred)
decision = lr.decision_function(trainScaled[:5])
print(np.round(decision, decimals=2))

# 이진 분류 sigmoid, 다중 분류 softmax
from scipy.special import softmax
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=2))
