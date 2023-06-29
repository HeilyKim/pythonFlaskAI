import matplotlib.pyplot as plt  #folder.file
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0,
                32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0,
                 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
length = bream_length+smelt_length
weight = bream_weight+smelt_weight
fishData = [[l,w] for l,w in zip(length,weight)]  #fish list에 [l,w]를 넣는데 [l,w]는 for문으로 l,w -> zip(length,weight)에서 꺼내옴
fishTartget = [1]*35+[0]*14  # bream:1 smelt:0
kn.fit(fishData,fishTartget) #fishData를 fishTarget으로 학습해라 =(fishData를 fishTarget(=도미,빙어 label)이랑 매치함)
print(kn.predict([[58,30],[30,600],[10,20]]))
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30, 600, marker='^') #predict할거 마커
plt.xlabel('length')
plt.ylabel('weight')
plt.legend(["bream 1", "smelt 0"])
#plt.show()
#학습된 데이터 확인용
# print(kn._fit_X)
# print(kn._fit_Y)

# kn49 = KNeighborsClassifier(n_neighbors=49) #default=5=근접 5개의 data를 근거로 하겠다
# kn49.fit(fishData, fishTartget)
# kn49.score(fishData, fishTartget) #정확도
