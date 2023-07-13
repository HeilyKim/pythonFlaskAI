#실제상 최근접 데이터의 평균을 출력하는거임(n_neighbour 수 따라감)
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
# x: 공부시간
# y: 점수
x = [3,10,20,30,40]
y = [60,70,80,90,100]
x = np.array(x)
x = x.reshape(-1,1)
knr = KNeighborsRegressor(n_neighbors=2)
knr.fit(x,y)
pred = knr.predict([[35]])
print(pred)