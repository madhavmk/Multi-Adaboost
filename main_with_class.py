__author__ = 'Xin'
'''
Reference:
Multi-class AdaBoosted Decision Trees:
http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html
'''

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_gaussian_quantiles
#from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
#from metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

from multi_AdaBoost import AdaBoostClassifier as Ada

'''
X, y = make_gaussian_quantiles(n_samples=13000, n_features=10,
                               n_classes=3, random_state=1)


print('X ',X)
print('X type',type(X))
print('y ',y)
print('y type',type(y))

n_split = 3000

X_train, X_test = X[:n_split], X[n_split:]
y_train, y_test = y[:n_split], y[n_split:]
'''
data = pd.read_csv("GALEX_data-extended-feats-original.csv")
#data = pd.read_csv("GALEX_data-extended-feats.csv")
print('data shape is ',data.shape)
print('data head \n',data.head(),'\n')

sc = StandardScaler()
X=sc.fit_transform(data.drop('class',axis=1))
X=data.drop('class',axis=1)
y= data['class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state=24)

print('X_train')
print(X_train)

print('converted X_train')
X_train=X_train.to_numpy()
X_test=X_test.to_numpy()
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()
#print(np_obj)




"""
bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=1000,
    learning_rate=0.1)


bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=1000,
    learning_rate=0.1,
    algorithm="SAMME")


bdt_real.fit(X_train, y_train)
bdt_discrete.fit(X_train, y_train)



n_trees_discrete = len(bdt_discrete)
n_trees_real = len(bdt_real)
"""



for lr in np.arange(0.05,1.05,0.05):
    for ne in [10,50,100,500,1000]:

        bdt_real_test = Ada(
            base_estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=ne,
            learning_rate=lr)
        bdt_real_test.fit(X_train, y_train)

        bdt_discrete_test = Ada(
            base_estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=ne,
            learning_rate=lr,
            algorithm='SAMME')
        bdt_discrete_test.fit(X_train, y_train)

        """
        discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
        real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
        """
        test_real_errors=bdt_real_test.estimator_errors_[:]
        test_discrete_errors=bdt_discrete_test.estimator_errors_[:]
        """
        plt.figure(figsize=(15, 5))
        plt.subplot(221)
        plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,
                "b", label='SAMME', alpha=.5)
        plt.legend()
        plt.ylabel('Error')
        plt.xlabel('Number of Trees')
        plt.ylim((.2,
                max(real_estimator_errors.max(),
                    discrete_estimator_errors.max()) * 1.2))
        plt.xlim((-20, len(bdt_discrete) + 20))

        plt.subplot(222)
        plt.plot(range(1, n_trees_real + 1), real_estimator_errors,
                "r", label='SAMME.R', alpha=.5,color='r')
        plt.legend()
        plt.ylabel('Error')
        plt.xlabel('Number of Trees')
        plt.ylim((.2,
                max(real_estimator_errors.max(),
                    discrete_estimator_errors.max()) * 1.2))
        plt.xlim((-20, len(bdt_discrete) + 20))

        plt.subplot(224)
        plt.plot(range(1, n_trees_real + 1), test_real_errors,
                "r", label='test_real', alpha=.5, color='b')

        plt.legend()
        plt.ylabel('Error')
        plt.xlabel('Number of Trees')
        plt.ylim((.2,
                max(real_estimator_errors.max(),
                    discrete_estimator_errors.max()) * 1.2))
        plt.xlim((-20, len(bdt_discrete) + 20))

        plt.subplot(223)
        plt.plot(range(1, n_trees_real + 1), test_discrete_errors,
                "r", label='test_discrete', alpha=.5)

        plt.legend()
        plt.ylabel('Error')
        plt.xlabel('Number of Trees')
        plt.ylim((.2,
                max(real_estimator_errors.max(),
                    discrete_estimator_errors.max()) * 1.2))
        plt.xlim((-20, len(bdt_discrete) + 20))
        """

        from sklearn.metrics import accuracy_score

        print('_______ ne ',ne,'__lr ',lr)
        #print(accuracy_score(bdt_real.predict(X_test),y_test))
        print(accuracy_score(bdt_real_test.predict(X_test),y_test))
        #print(accuracy_score(bdt_discrete.predict(X_test),y_test))
        print(accuracy_score(bdt_discrete_test.predict(X_test),y_test))
        print('____________')
