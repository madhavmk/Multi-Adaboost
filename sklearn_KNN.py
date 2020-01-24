import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

from timeit import default_timer as timer

data = pd.read_csv("GALEX_data-extended-feats.csv")
print('data shape is ',data.shape)
print('data head \n',data.head(),'\n')

X=data.drop('class',axis=1)
y= data['class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=24)
#"""
###  Unweighted KNN ###
for num in range(1,500,2):
    #print("KNN Classifier  n = ",num)
    neigh = KNeighborsClassifier(n_neighbors=num)
    neigh.fit(X_train, y_train)

    y_pred = neigh.predict(X_test)
    print(num,',',accuracy_score(y_test,y_pred))
    #print('accuracy score = ',accuracy_score(y_test,y_pred))
    #print(confusion_matrix(y_test,y_pred))
    #print(classification_report(y_test,y_pred))
###  ###
#"""

"""
###  Weighted KNN ###
for num in range(1,500,2):
    #print("KNN Classifier  n = ",num)
    neigh = KNeighborsClassifier(n_neighbors=num, weights='distance',metric='minkowski',p=2)
    neigh.fit(X_train, y_train)

    y_pred = neigh.predict(X_test)
    print(num,',',accuracy_score(y_test,y_pred))
    #print('accuracy score = ',accuracy_score(y_test,y_pred))
    #print(confusion_matrix(y_test,y_pred))
    #print(classification_report(y_test,y_pred))
###  ###
"""