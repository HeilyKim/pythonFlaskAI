from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
data = pd.read_csv('train.csv')
data['Sex'] = data['Sex'].map({'female':1,'male':0})
# print(data['Sex'])
# print(data.isnull().sum())
data['Age'].fillna(value = data['Age'].mean(),inplace=True) #NA를 mean 숫자로 대신함으로 결측치 제거
dummies = pd.get_dummies(data['Pclass'])
print(dummies)
del data['Pclass']
print(data.head())
data = pd.concat([data,dummies],axis=1,join='inner')
data.rename(columns = {1:'FirstClass',2:'SecondClass',3:'ThirdClass'},inplace=True)
data['FirstClass'] = data['FirstClass'].map({False:0,True:1})
data['SecondClass'] = data['SecondClass'].map({False:0,True:1})
data['ThirdClass'] = data['ThirdClass'].map({False:0,True:1})
input = data[['Age','Sex','FirstClass','SecondClass','ThirdClass']]
target = data['Survived']

trainX,testX,trainY,testY = train_test_split(input,target,random_state=42)
# print(trainX.shape)

ss = StandardScaler()
ss.fit(trainX)
trainScaled = ss.transform(trainX)
testScaled = ss.transform(testX)

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
lr = LogisticRegression()
sgd = SGDClassifier(loss='log')

lr.fit(trainScaled,trainY)
trainScore = lr.score(trainScaled,trainY)
testScore = lr.score(testScaled,testY)

print('LogisticR: ',trainScore,testScore)
print()

sgd.fit(trainScaled,trainY)
trainSGDScore = sgd.score(trainScaled,trainY)
testSGDScore = sgd.score(testScaled,testY)
print('SGD: ',trainSGDScore,testSGDScore)

sgd.partial_fit(trainScaled,trainY)
trainSGDScore = sgd.score(trainScaled,trainY)
testSGDScore = sgd.score(testScaled,testY)
print('SGD ParitalFit: ',trainSGDScore,testSGDScore)