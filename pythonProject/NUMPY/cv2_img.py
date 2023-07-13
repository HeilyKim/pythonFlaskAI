import numpy as np
import cv2
array = cv2.imread('Dark.png')
print(array.shape)
array[:100,:100] = [0,0,255]
array=array[:,:,:100]*1
cv2.imwrite('test.png',array)
array = np.random.randint(0,255,(150,150,3))
cv2.imwrite('aa.png',array)