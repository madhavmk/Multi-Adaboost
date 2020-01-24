    import pandas as pd
    import numpy as np

    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    import xgboost as xgb

    import pandas as pd
    import numpy as np

data = pd.read_csv("GALEX_data-extended-feats-original.csv")
print('data shape is ',data.shape)
print('data head \n',data.head(),'\n')

X=data.drop('class',axis=1)
y= data['class']



data_dmatrix = xgb.DMatrix(data=X,label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=24)
for lr in np.arange(0.05,1.05,0.05):
    for ne in [100,500,1000]:
        model=xgb.XGBClassifier(random_state=1,learning_rate=lr,n_estimators=ne)
        model.fit(X_train, y_train)
        print('num_estimators ,',ne,',',' learning rate ,',lr,',',model.score(X_test,y_test))