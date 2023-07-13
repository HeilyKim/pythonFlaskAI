import numpy as np
import matplotlib.pyplot as plt
array1 = np.random.randint(0,10,(3,3))
array2 = np.random.normal(0,10,(3,3)) #Draw random samples from a normal (Gaussian) distribution.
print(f'array1: \n{array1}')
print(f'array2: \n{array2}')
array3 = np.concatenate([array1,array2])
print(f'array3: \n{array3}')
# plt.imshow(array1)
# plt.show()

array4 = np.array([1,2,3,4,5,6,7,8,9,10])
array5 = array4.reshape(-1,2) #1-10까지 2배열로 끝까지 만들기
print(f'reshape(-1,2): \n{array5}')

array4 = np.array([1,2,3,4])
array5 = array4.reshape(2,2) #1-4끝까지 2*2로 만들기
print(f'reshape(2,2): \n{array5}')
