import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from array import *
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
fish_target = fish['Species'].to_numpy()
trainX,testX,trainY,testY = train_test_split(fish_input,fish_target,random_state=42)
ss = StandardScaler()
ss.fit(trainX)
trainScaled = ss.transform(trainX)
testScaled = ss.transform(testX)
print(trainScaled[:5])
print(testScaled[:5])

sgd = SGDClassifier(loss='log',max_iter=10,random_state=42)
sgd.fit(trainScaled,trainY)

score = sgd.score(trainScaled,trainY)
print(score)
score = sgd.score(testScaled,testY)
print(score)

sgd.partial_fit(trainScaled,trainY)
score = sgd.score(trainScaled,trainY)
print(score)
score = sgd.score(testScaled,testY)
print(score)
sgd = SGDClassifier(loss='log',random_state=42)
trainScore = []
testScore = []
classes = np.unique(trainY)
for _ in range(0,300):
    sgd.partial_fit(trainScaled,trainY,classes=classes)
    trainScore.append(sgd.score(trainScaled,trainY))
    testScore.append(sgd.score(testScaled,testY))
plt.plot(trainScore)
plt.plot(testScore)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()