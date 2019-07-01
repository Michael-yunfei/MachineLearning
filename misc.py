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
import random

print(sys.version)

# check working directory
os.getcwd()
os.chdir('/Users/Michael/Documents/MachineLearning/ELS')

# read prostate dataset and run regression with default package
prostate = pd.read_csv('prostate.csv', delim_whitespace=True)
prostate.info()
prostate.head()
prostate
prostate.columns
# Index(['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45',
# 'lpsa', 'train']

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(prostate.age, prostate.lweight)
plt.plot(prostate.age)

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


# convert mat file into csv
mat = scipy.io.loadmat('file.mat')
mat = {k:v for k, v in mat.items() if k[0] != '_'}
data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.iteritems()})
data.to_csv("example.csv")


alist = [1, 2, 3, 6]
alist[:]

blist = [[1], [2], [3]]
[n[0] for n in blist]

# generate random intger
random.sample(range(100), 50)
[i for i in range(10)] * (-1)

mma = np.asmatrix(np.array(range(18)).reshape(6, 3))
mma[[-5, -3, -1], :]
sample = mma
sample_length = 4
random_index = random.sample(range(sample.shape[0]), sample_length)
trainSample = sample[random_index, :]
testSample = np.delete(sample, random_index, 0)

n = input_x.shape[0]
theta = np.array([1.23, 2.89]).reshape(-1, 1)
fx = input_x @ theta
error = np.abs(input_y - fx)
global_epsilon = 1.13


def hl(element):
    global global_epsilon
    if element <= global_epsilon:
        loss = 1/2 * element**2
    else:
        loss = global_epsilon * element - 1/2 * global_epsilon**2

    return(loss)


hlvector = np.vectorize(hl)
hlvector(error)


va = np.array(range(9)).reshape(3, 3)
# array([[0, 1, 2],
#        [3, 4, 5],
#        [6, 7, 8]])

vb = np.array(range(9)).reshape(3, 3) - 2
# array([[-2, -1,  0],
#        [ 1,  2,  3],
#        [ 4,  5,  6]])


np.asmatrix(va)[:, 1:]

va[2:3, :]

np.asmatrix(va)[:, 0]


np.random.rand(3, 1)

int(range(10))

va - vb

# array([[2, 2, 2],
#        [2, 2, 2],
#        [2, 2, 2]])


vc = []

vc.append(vb)

np.stack(vb, axis=1)


for i in range(10):
    print(i)

hello = np.array(range(10))
len(hello)


hello[hello == 6]
np.where(hello==6)

va + np.array([3]).reshape(-1, 1)


dict1 = {'a':[[1, 2], [2, 3]], 'b':[[3, 6], [5, 9]]}

dict_sum = np.zeros([2, 2])
for i in dict1:
    dict_sum += np.asmatrix(dict1[i])












# End of code
