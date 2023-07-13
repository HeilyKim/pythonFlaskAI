import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate,StratifiedKFold,GridSearchCV
import numpy as np

wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(data, target, random_state=42)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_input, train_target)
dt.fit(test_input,test_target)

print(dt.score(train_input, train_target))
print(dt.score(test_input,test_target))

# pred = dt.predict([[9.4,2.5,6.61],[5.6,19,3.56]])
# print(pred)
#
# crossValScore = cross_validate(dt,train_input,train_target)
# print(crossValScore)
#
# myFold = StratifiedKFold(n_splits=10,random_state=42,shuffle=True)
# scores = cross_validate(dt,train_input,train_target,cv=myFold)
# print(scores)

#parameter tunning
params = {'min_impurity_decrease':[0.0001,0.0002,0.0003,0.0004,0.0005],'max_depth':range(3,10)}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
dt = gs.best_estimator_
print(dt.score(train_input, train_target))
print(gs.best_params_)
print(gs.cv_results_['mean_test_score'])
best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])

params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
          'max_depth': range(5, 20, 1),
          'min_samples_split': range(2, 100, 10)
          }
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
print(gs.best_params_)
print(np.max(gs.cv_results_['mean_test_score']))