from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
# acls: -4~0,5~10 bcls: 2~6,0~5 ccls: -8~-4 -7~5 -2~5
acls = np.array([[x,y]
                for x,y in zip(np.random.randint(-4,0,(10)),np.random.randint(5,10,(10)))])
bcls = np.array([[x,y]
                for x,y in zip(np.random.randint(2,6,(10)),np.random.randint(0,5,(10)))])
ccls = np.array([[x,y]
                for x,y in zip(np.random.randint(-8,-4,(10)),np.random.randint(-7.5,-2.5,(10)))])
print(acls)
print(bcls)
print(ccls)
target = ['A']*10+['B']*10+['C']*10
data = np.concatenate([acls,bcls,ccls])
lr=LogisticRegression()
lr.fit(data,target)
pred = lr.predict([[5,6],[5,10]]) #[x][y]
print(pred)
plt.scatter(acls[:,0],acls[:,1])
plt.scatter(bcls[:,0],bcls[:,1])
plt.scatter(ccls[:,0],ccls[:,1])
plt.scatter([5,5],[6,10]) #[x,y]
plt.show()