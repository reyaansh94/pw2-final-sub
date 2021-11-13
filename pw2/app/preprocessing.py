from pandas.core.frame import DataFrame
import numpy as np
from numpy import nan
import matplotlib.pyplot as plt
import pandas as pd
import sklearn 

def numcat(data1: DataFrame):
    numerical_cols = data1.select_dtypes(include=['number'])
    categorical_cols = data1.select_dtypes(include=['object'])
    numerical_cols = numerical_cols.apply(lambda x: x.fillna(x.mean()), axis=0)
    categorical_cols = categorical_cols.apply(lambda x: x.fillna(x.value_counts().index[0]))
    data1 = pd.concat([categorical_cols, numerical_cols], axis=1)
    return data1