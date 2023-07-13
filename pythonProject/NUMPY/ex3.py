import numpy

a = [10,20,30]
b = [40,50,60]
c = a+b
print(f'numpy를 안 쓰면: {c}')
import numpy as np
a = np.array(a)
b = np.array(b)
c = a+b
print(f'numpy를 쓰면: {c}')

#0-15 4*4만들고 10보다크면 100으로 교체
array = np.arange(16).reshape(4,4)
indexes = array>10
array[indexes] = 100
print(array)

#0-15 4*4의 열(2차원 axis=0) sum 하기
array = np.arange(16).reshape(4,4)
value = np.sum(array,axis=0)
print(value)

#0-15 4*4의 행(2차원 axis=1) sum 하기
array = np.arange(16).reshape(4,4)
value = np.sum(array,axis=1)
print(value)

# array저장 및 부르기
np.save('a.npy',[71,62,53,34,25,36])
data = numpy.load('a.npy')
print(data)

a = [10,20,30]
b = a
a[0] = 50
print(a)
print(b)
a = np.array(a)
b = a.copy()
a[1] = 99
print(a)
print(b)