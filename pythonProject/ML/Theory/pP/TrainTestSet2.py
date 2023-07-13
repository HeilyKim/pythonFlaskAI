from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
               31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
               35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
               10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
               500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
               700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
               7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

a = np.column_stack(([1, 2, 3], [4, 5, 6]))  # [[1 4][2 5][3 6]]
fish = np.column_stack((fish_length, fish_weight))
t = np.ones(3)  # [1 1 1] 1 3개 만들어달라
t1 = np.zeros((3, 3))  # 0으로 3*3만들어달라
tfull = np.full((2, 2), 10)  # 10으로 2*2 만들어줘
fishTarget = np.concatenate((np.ones(35), np.zeros(14)))  # np.concatenate(a,b): a랑 b나열
train_input, test_input, train_target, test_target = train_test_split(fish, fishTarget, stratify=fishTarget,
                                                                      random_state=42)
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
score = kn.score(test_input, test_target)
pred = kn.predict([[25, 150]])  # <<수상한 도미
# print(score)
# print(pred)
# plt.scatter(train_input[:, 0], train_input[:, 1])
# plt.scatter(25, 150, marker='^')  # 수상산 도미 표기
# distances, indexes = kn.kneighbors([[25, 150]])  # 25,150과 가까운 이웃들 표시
# print('거리 = ', distances)
# print('순번 = ', indexes)
# print(train_input[indexes])
# plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
# plt.title('fish_train')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()
mean = np.mean(train_input, axis=0)  # train data가 기준임
std = np.std(train_input, axis=0)
# 표준편자로인한 스케일 맞추기(수상한 도미쓰)
ss = StandardScaler()
ss.fit(train_input, train_target)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
strangeDomi = ss.transform([[25, 150]])
plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(strangeDomi[:, 0], strangeDomi[:, 1], marker="^")
plt.show()
kn.fit(train_scaled, train_target)
print()

pred = kn.predict(strangeDomi)
print(pred)
