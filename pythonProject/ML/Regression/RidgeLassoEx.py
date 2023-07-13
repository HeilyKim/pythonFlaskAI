from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures # 항을 늘리는 라이브러리?
import numpy as np

# [2,5]가 들어오면 10이 나오고 [3,6]이 들어오면 18이 나온다.
train_data = np.array([[2,5], [3,6], [4,8], [5,10], [6,13]])
test_data = np.array([[3,7], [2,9],[5,11]])
target = np.array([10, 18, 32, 50, 78])
test_target = np.array([21, 18, 55])

lr = LinearRegression()
lr.fit(train_data, target) # 학습을 해라

predTrain = lr.predict(train_data) # 예측하라.
predTest = lr.predict(test_data)

print(predTrain)
print(predTest)

poly = PolynomialFeatures(include_bias=False, degree=5) # 제곱항이 나온다. (include_bias=False) -> 1이 안나온다. (degree=5) -> 5항을 더 늘려준다.
poly.fit(train_data)
train_poly = poly.transform(train_data)
test_poly = poly.transform(test_data)
print(train_poly)

lr = LinearRegression()
lr.fit(train_poly, target)

predTrain = lr.predict(train_poly)
predTest = lr.predict(test_poly)

print(predTrain)
print(predTest)

# 위처럼 항을 늘리면 예측데이터랑 똑같이 나왔다.이런한 경우를 훈련데이터에 과대적합되었다라고 한다. 또 다른 말로 오버피팅되었다라고 한다.
# 과대적합이 되면 규제가 없어져 오류가 발생이 된다. 다시 말해 오버피팅이 되면 그래프가 0과 거의 근접하다. 이를 막기위해 릿지 회귀를 사용한다.

ss = StandardScaler()
ss.fit(train_poly)

train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

ridge = Ridge(alpha=0.001)
ridge.fit(train_poly, target)

predTrain = lr.predict(train_poly)
predTest = lr.predict(test_poly)

print(predTrain)
print(predTest)