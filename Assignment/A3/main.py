# Assignment 3 - Classification
# @ Coco, Lada, Michael

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import os

# check working directory
os.getcwd()
os.chdir('/Users/Michael/Documents/MachineLearning/Assignment/A3')

###############################################################################
# Part I - Load the Dataset, select two types (classes) of wine.
###############################################################################
indexNames = ['WineClass', 'Alcohol', 'Malic acid', 'Ash',
              'Alcalinity of ash', 'Magnesium',
              'Total phenols', 'Flavanoids',
              'Nonflavanoid phenols',
              'Proanthocyanins', 'Color intensity',
              'Hue', 'OD280/OD315 of diluted wines',
              'Proline']
len(indexNames)

wine = pd.read_csv('wine.csv', names=indexNames)
wine.shape  # (178, 14)
wine.head()

# check whether dataset is balanced or not:
# If it is (perfectly) balanced, each class of Y should have the same
# number of observations. This is perfect for training our model
# If it is not perfectly balanced, we have to be careful: e.g.
# if Y=0 has 50 observations, Y=1 has 20 observations, Y=2 has 10 observations
# What should we do for this case?

uniqueY = np.unique(wine.WineClass)  # get unique values
len(uniqueY)  # 3
for i in range(len(uniqueY)):
    print(wine[wine.WineClass == uniqueY[i]].shape)

# (59, 14)
# (71, 14)
# (48, 14)
# The dataset is not prefectly balanced, but the differences of number of
# observations for three classes are not that big.

# Now we select only classes 1, 3 for this part and feature'Proanthocyanins'.
# In this binary classification exercise assign label 0 to Class 1 and
# label 1 to Class 3.

wineSubset = wine.query('WineClass==1| WineClass==3')  # filter 1 and 3 out
wineSubset = wineSubset[['WineClass', 'Proanthocyanins']]
wineSubset['newclass'] = np.where(wineSubset['WineClass'] == 1, 0, 1)
wineSubset = wineSubset[['newclass', 'Proanthocyanins']]
wineSubset.shape
wineSubset.head()

###############################################################################
# Part II: Binary Classification with One Feature
###############################################################################




























# End of code
