import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
wine = pd.read_csv('https://bit.ly/wine_csv_data')
# print(wine.head())
# print(wine.info())
# print(wine.isnull().sum())
# print(wine.describe()) #statistic info

input = wine[['alcohol','sugar','pH']]
target = wine['class']
# print(input.shape)
# print(target.shape)

trainX,testX,trainY,testY = train_test_split(input,target,random_state=42)
# print(trainX.shape)
# print(testX.shape)

ss = StandardScaler()
ss.fit(trainX)
trainScaled = ss.transform(trainX)
testScaled = ss.transform(testX)
# print(trainScaled[:5])
# print(testScaled[:5])

lr = LogisticRegression()
lr.fit(trainScaled,trainY)
trainScore = lr.score(trainScaled,trainY)
testScore = lr.score(testScaled,testY)
print(f'LR-train score: {trainScore} test score: {testScore}')

valiData = [[9.0,14.9,3.13],[10.6,6.2,3.6],[9.5,8.5,3.32]]
valiData = ss.transform(valiData)
pred = lr.predict(valiData)
#print(pred)
dt = DecisionTreeClassifier(min_impurity_decrease=0.0005, random_state=42)
dt.fit(trainScaled,trainY)
trainDTScore = dt.score(trainScaled,trainY)
testDTScore = dt.score(testScaled,testY)
print(f'DT-train score: {trainDTScore} test score: {testDTScore}')

plt.figure(figsize=(10,7))
plot_tree(dt,max_depth=2,filled=True,feature_names=['alcohol','sugar','pH'])
#plt.show()
print(dt.feature_importances_)