from pandas.core.frame import DataFrame
import numpy as np
from numpy import nan
import matplotlib.pyplot as plt
import pandas as pd
import sklearn 
from preprocessing import numcat
import joblib


def prediction (data1: DataFrame):
    data1 = numcat (data1 )
    X = data1['MasVnrArea'].values.reshape(-1,1)
    load_model = joblib.load('C:/Users/goldu/Downloads/pw2/models/regression.joblib' )
    return load_model.predict( X)