# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:05:06 2019

@author: ajseshad
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import pandas as pd
 
def ReadDataset(filename):
    dataset = pd.read_csv(filename, usecols = [2,5])
    #dataset['Timestamp'] = pd.to_datetime(dataset['Lens_IngestionTime'], infer_datetime_format=True)
    #return dataset.set_index(['Timestamp'])
    return dataset

def PlotDataset(dataset):
    plt.xlabel('Day')
    plt.ylabel('Hosted Service Count')
    plt.plot(dataset)

def linearRegression(X, y):
    # Fitting Linear Regression to the dataset
    from sklearn.linear_model import LinearRegression
    linReg = LinearRegression()
    linReg.fit(X, y)
    # Visualising the Linear Regression results
    plt.scatter(X, y, color = 'red')
    plt.plot(X, linReg.predict(X), color = 'blue')
    plt.title('Linear Regression')
    plt.xlabel('Time')
    plt.ylabel('Hosted Service Count')
    plt.show()
    
dataset = ReadDataset("SubscriptionTimeSeries.csv")
#X = np.arange(len(dataset))
X = dataset['Index']
X = np.reshape(X.values, (-1, 1))
y = dataset.iloc[:, 1]
y = np.reshape(y.values, (-1, 1))
linearRegression(X,y)
