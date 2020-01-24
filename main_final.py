import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_gaussian_quantiles

from sklearn.metrics import accuracy_score, classification_report

from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


data = pd.read_csv("GALEX_data-extended-feats-original.csv")
#data = pd.read_csv("GALEX_data-extended-feats.csv")
print('data shape is ',data.shape)
print('data head \n',data.head(),'\n')

sc = StandardScaler()
X=sc.fit_transform(data.drop('class',axis=1))
X=data.drop('class',axis=1)
y= data['class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=24)

print('X_train')
print(X_train)

print('converted X_train')
X_train=X_train.to_numpy()
X_test=X_test.to_numpy()
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()


from multi_AdaBoost import AdaBoostClassifier as Ada

lr=0.05
ne=100

bdt_real_test = Ada(
    base_estimator=DecisionTreeClassifier(max_depth=2),
    n_estimators=ne,
    learning_rate=lr)
bdt_real_test.fit(X_train, y_train)

bdt_discrete_test = Ada(
    base_estimator=DecisionTreeClassifier(max_depth=2),
    n_estimators=ne,
    learning_rate=lr,
    algorithm='SAMME')
bdt_discrete_test.fit(X_train, y_train)

test_real_errors=bdt_real_test.estimator_errors_[:]
test_discrete_errors=bdt_discrete_test.estimator_errors_[:]

from sklearn.metrics import accuracy_score

print('_______ ne ',ne,'__lr ',lr)
#print(accuracy_score(bdt_real.predict(X_test),y_test))
#print(accuracy_score(bdt_real_test.predict(X_test),y_test))
#print(accuracy_score(bdt_discrete.predict(X_test),y_test))
print("Accuracy")
print(accuracy_score(bdt_discrete_test.predict(X_test),y_test))
print('____________')
print(classification_report(bdt_discrete_test.predict(X_test),y_test, digits=4))