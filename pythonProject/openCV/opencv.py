from sklearn.linear_model import LogisticRegression
import cv2
import numpy as np
a = cv2.imread('a.png')
b = cv2.imread('b.png')
c = cv2.imread('c.png')
abc = np.concatenate([a,b,c])
abc = abc.reshape(3,-1)
print(abc.shape)

lr = LogisticRegression()
lr.fit(abc,['A','B','C'])

test = cv2.imread('test.png')
test = test.reshape(1,-1)
pred = lr.predict(test)
print(pred)