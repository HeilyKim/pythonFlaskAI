from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import cv2
apple = cv2.imread('img.png',cv2.IMREAD_GRAYSCALE)
print(apple.shape)

apple = cv2.resize(apple,(100,100))
print(apple.shape)
fruits = np.load('fruits_300.npy')
# print(fruits.shape)
fruits_2d = fruits.reshape(-1, 10000)  # (300, 10000) 10000개의 특징이 있는 300개의 데이터
# print(fruits_2d.shape)

# plt.imshow(fruits[0], cmap='gray_r')
# plt.show()
# plt.imshow(fruits[0]-np.mean(fruits[:100],axis=(1,2)), cmap='gray_r')
# plt.show()

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)
# print(km.labels_)
print(np.unique(km.labels_, return_counts=True))


# rows = 3
# a = 3 if rows < 2 else 10  # if rows <2 then a = 3 else a=10
# print(a)


def draw_fruits(arr, ratio=1):
    n = len(arr)
    rows = int(np.ceil(n / 10))  # ceil 올림처리
    cols = n if rows < 2 else 10
    _, axs = plt.subplots(rows, cols, figsize=(cols, rows), squeeze=False)  # squeeze -> [a,b]가됨 안하면 안됨
    for i in range(rows):
        for j in range(cols):
            if (i * 10 + j) < n:
                axs[i, j].imshow(arr[i * 10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()


# draw_fruits(fruits[km.labels_ == [0]])

# kmeans으로 cluster 평균값 그리기
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)

# 각 이미지 평균값으로 그리기
appleMean = np.mean(fruits_2d[0:100], axis=0)
pineappleMean = np.mean(fruits_2d[100:200], axis=0)
bananaMean = np.mean(fruits_2d[200:300], axis=0)
means = []
means.append(appleMean)
means.append(pineappleMean)
means.append(bananaMean)
means = np.array(means)
means = means.reshape(-1, 100, 100)
print(means.shape)

draw_fruits(means, ratio=3)

pred = km.predict(apple.reshape(-1,10000))
print(pred)

inertia = []
def doPrint():
    for k in range(2,7):
        km = KMeans(n_clusters=k,random_state=42)
        km.fit(fruits_2d)
        inertia.append(km.inertia_)
    plt.plot(range(2,7),inertia)
    plt.show()
doPrint()