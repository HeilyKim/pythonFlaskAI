import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans

fruits = np.load('fruits_300.npy')
print(fruits.shape)
fruits_2d = fruits.reshape(-1, 100 * 100)
print(fruits_2d)

##주성분 찾기
pca = PCA(n_components=50)
pca.fit(fruits_2d)
print('주성분:', pca.components_.shape)  # 주성분이 50개 나옴

pca50 = pca.components_.reshape(-1, 100, 100)
fig, axs = plt.subplots(5, 10, squeeze=False)
for i in range(5):
    for j in range(10):
        axs[i, j].imshow(pca50[i * 10 + j], cmap='gray_r')
        axs[i, j].axis('off')
# plt.show()

##차원 축소
fruits_pca = pca.transform(fruits_2d)
print('fruits_2d:', fruits_2d.shape)
print('fruits_pca:', fruits_pca.shape)

##차원축소 원상복귘ㅋㅋㅋㅋ
fruits_inverse = pca.inverse_transform(fruits_pca)
print('fruits_inverse:', fruits_inverse.shape)

# 이미지로 그리려면 100*100px 로 봐꿔야됨
fruits_inverse = fruits_inverse.reshape(-1, 100, 100)
print(fruits_inverse.shape)

# fig, axs = plt.subplots(10, 10, squeeze=False)
# for i in range(10):
#     for j in range(10):
#         axs[i, j].imshow(fruits_inverse[i * 10 + j], cmap='gray_r')
#         axs[i, j].axis('off')
# plt.show()
#
# fig, axs = plt.subplots(10, 10, squeeze=False)
# for i in range(10):
#     for j in range(10):
#         axs[i, j].imshow(fruits_inverse[(i + 10) * 10 + j], cmap='gray_r')
#         axs[i, j].axis('off')
# plt.show()
#
# fig, axs = plt.subplots(10, 10, squeeze=False)
# for i in range(10):
#     for j in range(10):
#         axs[i, j].imshow(fruits_inverse[(i + 20) * 10 + j], cmap='gray_r')
#         axs[i, j].axis('off')
# plt.show()

lr = LogisticRegression()
target = ['사과'] * 100 + ['파인애플'] * 100 + ['바나나'] * 100
scores = cross_validate(lr, fruits_2d, target)
print('test score:',np.mean(scores['test_score']))
print('fit time:',np.mean(scores['fit_time']))
scores = cross_validate(lr, fruits_pca, target)
print('test score:',np.mean(scores['test_score']))
print('fit time:',np.mean(scores['fit_time']))

pca = PCA(n_components=0.5)
pca.fit(fruits_2d)
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

scores =cross_validate(lr,fruits_pca,target)
print('test score:',np.mean(scores['test_score']))
print('fit time:',np.mean(scores['fit_time']))
plt.show()

km = KMeans(n_clusters=3,random_state=42)
km.fit(fruits_pca)
for label in range(0,3):
    data = fruits_pca[km.labels_ == label]
    plt.scatter(data[:,0],data[:,1])
plt.legend(['apple','banana','pipeapple'])
plt.show()
