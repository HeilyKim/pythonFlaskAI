import cv2
import matplotlib.pyplot as plt
origin_dark = cv2.imread('Dark.png',cv2.IMREAD_COLOR)
print(origin_dark.shape)
print(origin_dark[100,100]) #->100,100좌표의 색은[  0 195 219](G,B,R)
origin_dark[80:120,80:120] = [204,204,255]

roi = origin_dark[30:60,100:120] #이 좌표의 그림을
origin_dark[0:30,0:20] = roi #여기 좌표로 바꿈
plt.imshow(cv2.cvtColor(origin_dark,cv2.COLOR_BGR2RGB))
plt.show()