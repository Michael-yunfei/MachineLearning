# miscellaneous code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sklearn.linear_model as skl_lm
from sklearn import preprocessing
from sklearn.preprocessing import scale
import sys


print(sys.version)

# check working directory
os.getcwd()
os.chdir('/Users/Michael/Documents/MachineLearning/ELS')

# read prostate dataset and run regression with default package
prostate = pd.read_csv('prostate.csv', delim_whitespace=True)
prostate.info()
prostate.head()
prostate.columns
# Index(['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45',
# 'lpsa', 'train']

# preprare the data
pro_train = prostate[prostate.train == 'T']
pro_train.describe()
pro_x = pro_train.loc[:, 'lcavol':'pgg45'].astype(float)  # convert to float
pro_x.shape  # check the dimension
pro_y = pro_train.loc[:, 'lpsa']

# scale predictors(independent variables) to have unit variace
pro_xsd = scale(pro_x, with_std=True)  # this step is very common
pro_xsd.shape
pro_mlr = skl_lm.LinearRegression().fit(pro_xsd, pro_y)

# print results
print(np.around(pro_mlr.intercept_, 4))  # 2.4523
print(np.around(pro_mlr.coef_, 4))
# [ 0.711   0.2905 -0.1415  0.2104  0.3073 -0.2868 -0.0208  0.2753]
# do regression in sklearn is not very beautiful
# maybe you can try statsmodels OLS fit


# try matrix multiplication
a = np.array(range(1, 10)).reshape(3, 3)
b = np.ones([3, 3])
c = a @ b
np.transpose(c)


# try to define a function with keywords arguments
def mmscale(a, normalized=True):
    if normalized is True:
        a += 1
        return a
    else:
        return a


mmscale(4)  # 5
mmscale(4, normalized=False)  # 4


# normalize function
def normalize(array):
    arrayNorm = (array - array.min())/(array.max() - array.min())
    return arrayNorm


# standardlize function
def standard(array):
    arrayStand = (array - array.mean())/array.std()
    return arrayStand


np.apply_along_axis(normalize, axis=0, arr=a)
np.apply_along_axis(standard, axis=0, arr=a)
np.apply_along_axis(standard, axis=0, arr=pro_x)  # the result is same


vector_a = np.array(range(1, 9)).reshape([8, 1])
vector_b = np.array(range(2, 10)).reshape([8, 1])
vector_a * vector_b

np.sqrt(a)
np.linalg.inv(a)
a.diagonal()

vector_a**2


def mtnorm(matirx):
    def normalize(array):
        arrayNorm = (array - array.min())/(array.max() - array.min())
        return arrayNorm
    temp = np.apply_along_axis(normalize, axis=0, arr=matirx)
    return temp


matrix_a = np.ones([6, 6])
vector_a1 = np.zeros([6, 1])
np.hstack([vector_a1, matrix_a])


pro_x.columns
columnslist = ['intercept']
columnslist.extend(pro_x.columns.values)
columnslist


class Klass(object):

    @staticmethod  # use as decorator
    def statfunc(x):
        x += 5
        return x

    # _ANS = stat_func.__func__()  # call the staticmethod

    def method(self, h):
        ret = self.__class__.statfunc(h)
        return ret


class_a = Klass()
class_a.method(5)


class ACk():

    def __init__(self, a):
        self.a = a

    @staticmethod
    def normalize(array):
        arrayNorm = (array - array.min())/(array.max() - array.min())
        return arrayNorm

    # standardlize function
    @staticmethod
    def standard(array):
        arrayStand = (array - array.mean())/array.std()
        return arrayStand

    def doitdo(self):
        self.a = np.apply_along_axis(ACk.normalize, axis=0, arr=self.a)
        return self.a


class_b = ACk(a)
class_b.doitdo()











# End of code
