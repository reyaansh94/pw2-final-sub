
from pandas.core.frame import DataFrame
import numpy as np
from numpy import nan
import matplotlib.pyplot as plt
import pandas as pd
import sklearn 
from preprocessing import numcat
import joblib

def model_training(data1: DataFrame):
    data1 = numcat (data1 )
    X = data1['MasVnrArea'].values.reshape(-1,1)
    y = data1['SalePrice'].values.reshape(-1,1)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
    from sklearn.linear_model import LinearRegression
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    joblib.dump(regr, 'C:/Users/goldu/Downloads/pw2/models/regression.joblib')
    print('Coefficients: ', regr.coef_[0][0])
    print('Intercept: ', regr.intercept_[0])
    y_pred = regr.predict(X_test)
    print("Root mean square error (RMSE): %.2f" % np.sqrt(np.mean((y_pred - y_test) ** 2)))
    return (  np.sqrt(np.mean((y_pred - y_test) ** 2)))