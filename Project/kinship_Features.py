# ML - Project
# Kinship recognization
# @ Michael

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mtmg
from scipy import signal
import seaborn as sns
import numpy as np
from scipy.optimize import minimize
import random
import os
from pathlib import Path
from glob import glob
import sys
import math
import scipy.stats as stats
import scipy.io as spo  # for loading matlab file
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize

os.chdir('/Users/Michael/Documents/MachineLearning/Project')
###############################################################################
# Secton 1 - Understand the data strucutre
###############################################################################

# the folder - train, includes thousands of subfolders which inlcude the family
# relationships.
# Those kinships are stored in a csv file - train_relationship.csv
train_relat = pd.read_csv('train_relationships.csv')
train_relat.shape
train_relat.head()
# p1	p2
# 0	F0002/MID1	F0002/MID3
# 1	F0002/MID2	F0002/MID3
# 2	F0005/MID1	F0005/MID2
# 3	F0005/MID3	F0005/MID2
# 4	F0009/MID1	F0009/MID4
train_relat.shape

# where p1 and p2 indicate the relationships
f1 = mtmg.imread('./train/F0002/MID1/P00009_face3.jpg')
f3 = mtmg.imread('./train/F0016/MID1/P00162_face1.jpg')
plt.imshow(f1)
plt.imshow(f3)

# in this example, f1-f3 are kin (father-daughter);
# f2-f3 are kin (mother-daughter);
# no realtionship between f5 with f1 to f3;

f1.shape  # (224, 224, 3)

# you can see the third dimension is related to the colors of picture
plt.imshow(f1[:, :, 0])
plt.imshow(f1[:, :, 1])
plt.imshow(f1[:, :, 2])

plt.imshow(f3[:, :, 0])
plt.imshow(f3[:, :, 1])
plt.imshow(f3[:, :, 2])

# Let's check the second dimension
plt.imshow(f1[:, 1:100, :])
plt.imshow(f3[:, 1:100, :])

# second dimension works for the horizontal direction
# they face structure looks similar

# let's check the first dimension
plt.imshow(f2[1:100, :, :])
plt.imshow(f3[1:100, :, :])


###############################################################################
# Section 2 - features construction
###############################################################################

# read pictures in one-shot way
# read pictures based on the 'train_relationships.csv' file

train_relat = pd.read_csv('train_relationships.csv')
train_relat.head()
# p1	p2
# 0	F0002/MID1	F0002/MID3
# 1	F0002/MID2	F0002/MID3
# 2	F0005/MID1	F0005/MID2
# 3	F0005/MID3	F0005/MID2
# 4	F0009/MID1	F0009/MID4

# change the path in your laptop
path = '/Users/Michael/Documents/MachineLearning/Project/train/'
faces = {}  # initialize the dictionary to store the data
# read 100 pairs from train_relationships.csv
for i in range(train_relat.shape[0]):
    face1 = train_relat.iloc[i, 0]
    face2 = train_relat.iloc[i, 1]
    path1 = path+face1
    path2 = path+face2
    for m, file in enumerate(Path(path1).glob('*.jpg')):
        # only read the first one
        if m == 0:
            image1 = mtmg.imread(file)
    faces[face1] = image1

    for m, file in enumerate(Path(path2).glob('*.jpg')):
        # only read the first one
        if m == 0:
            image2 = mtmg.imread(file)
    faces[face2] = image2


# Python will just overwrite the value of the duplicate keys
faces.keys()
len(faces.keys())  # 2412
faces['F0002/MID1'][:, :, 2]

# Now, we transfer the matrix into 2 dimension
faces_2d = {}
for m, n in enumerate(faces):
    faces_2d[n] = faces[n][:, :, 2]

# # top 200 eigenvalues
faces_eig = np.zeros([len(faces_2d.keys()), 200])
for m, n in enumerate(faces_2d):
    fx_eig, ev = np.linalg.eig(faces_2d[n] @ faces_2d[n].T)
    fx_eig.sort()
    faces_eig[m, :] = fx_eig[24:]

faces_2d.keys()
faces_eig.shape
# save results to csv file
np.savetxt('faces_eig.csv', faces_eig,  delimiter=',')

###############################################################################
# Section 3 - creat the difference matrix
###############################################################################

# based on the eigenvalues, creat the differences of element wise

np.subtract(faces_eig[0, :], faces_eig[1, :]).shape

kin_features_diff = np.zeros([train_relat.shape[0], 200])
for i in range(train_relat.shape[0]):
    pair1 = train_relat.iloc[i, 0]
    pair2 = train_relat.iloc[i, 1]
    loc1 = np.where(np.asarray(list(faces.keys())) == pair1)[0]
    loc2 = np.where(np.asarray(list(faces.keys())) == pair2)[0]
    featrues_diff = np.subtract(faces_eig[loc1[0], :], faces_eig[loc2[0], :])
    kin_features_diff[i, :] = featrues_diff

kin_features_diff.shape
kin_features_diff[0, :]
np.linalg.norm(kin_features_diff[0, :])  #368.85248204181164

# create the balanced data - 3598 nonkinship

random_pairs1 = np.random.randint(0, 2412, 5000)
random_pairs2 = np.random.randint(0, 2412, 5000)
nonkin_featrues_diff =  np.zeros([train_relat.shape[0], 200])
for i in range(train_relat.shape[0]):
    pair1 = random_pairs1[i]
    pair2 = random_pairs2[i]
    if np.abs(pair1 - pair2) <= 4:
        pair2 = random_pairs2[i+1]
        if np.abs(pair1 - pair2) <= 4:
            print(i)
    featrues_diff = np.subtract(faces_eig[pair1, :], faces_eig[pair2, :])
    nonkin_featrues_diff[i, :] = featrues_diff

nonkin_featrues_diff.shape  # (3598, 200)
nonkin_featrues_diff

# combine the data to have the balanced one
kin_labels = np.ones([3598, 1])
nonkin_labels = np.zeros([3598, 1])

kin_mat = np.hstack([kin_features_diff, kin_labels])
nonkin_mat = np.hstack([nonkin_featrues_diff, nonkin_labels])

data_mat = np.vstack([kin_mat, nonkin_mat])

data_mat.shape

# save it to csv
np.savetxt('data_mat.csv', data_mat, delimiter=',')

###############################################################################
# Section 4 - Visualize the features
###############################################################################

# os.chdir('/Users/Michael/Documents/MachineLearning/Project')

# Load the dataset
data_frame = pd.read_csv('data_mat.csv', header=None)
data_frame.shape
data_mat = np.asmatrix(data_frame)

# test the data
trainX = data_frame.iloc[:, 0:200]
trainY = data_frame.iloc[:, -1]

# plot the mean of features
plt.style.use('ggplot')
tx = np.linspace(1, 201, 200)
plt.plot(tx, np.mean(trainX.iloc[0:3598, :], axis=0),
         label='Mean of kinship features')
plt.plot(tx, np.mean(trainX.iloc[3598:, :], axis=0),
         label='Mean of nonkinship features')
plt.legend()
plt.savefig('meanfeatures.png', dpi=600, bbox_inches='tight')


# plot the histograms of features
features_Class_1 = np.array(data_frame[data_frame.iloc[:, -1]==1].iloc[:, 90])
features_Class_0 = np.array(data_frame[data_frame.iloc[:, -1]==0].iloc[:, 90])
plt.hist(features_Class_1,density=True,label='Kinship Feature',alpha = 0.7,
         bins=np.arange(min(features_Class_1)-0.3,
                        max(features_Class_1)+0.3,0.3))
plt.hist(features_Class_0,density=True,label='Nonkship Feature',alpha = 0.7,
         bins=np.arange(min(features_Class_0)-0.3,
                        max(features_Class_0)+0.3,0.3))

plt.legend(loc='upper right')
plt.savefig('histfeatures.png', dpi=600, bbox_inches='tight')

# plot the histograms of distances

data_norm = np.zeros([data_mat.shape[0], 1])
for i in range(data_mat.shape[0]):
    data_norm[i, :] = np.linalg.norm(data_mat[i, 0:200])

con1 = np.where(data_norm <=600)[0]

plt.hist(np.array(data_norm[con1][0:3570]),
         density=True,label='Kinship Distance',alpha = 0.7,
         bins=300)
plt.hist(np.array(data_norm[con1 ][3570:]),
         density=True,label='Nonkship Distance',alpha = 0.7,
         bins=300)

plt.legend(loc='upper right')
plt.savefig('histdistance.png', dpi=600, bbox_inches='tight')


###############################################################################
# More on properties of features
###############################################################################

# This part is to calcuate the probabilities
# kin_eigven = pd.read_csv('kin_dataset.csv', header=None)
#
# kin_eigven
#
# kin_pair = kin_eigven.iloc[0:2412, 0:200]
#
# # calculate the corrleation
# # randome select 100 individuals
# kin_pair_rdm = np.random.randint(0, kin_pair.shape[0], 100)
# kin_pair_mat = np.asmatrix(kin_pair.iloc[kin_pair_rdm, :])
#
# kin_pair_cor0 = np.zeros([100*99, 1])
# kin_pair_cor1 = np.zeros([99, 1])
# for m in range(100):
#     sel_one = kin_pair_mat[m, :]
#     sel_others = np.delete(kin_pair_mat, m, axis=0)
#     for n in range(99):
#         kin_pair_cor1[n] = np.corrcoef(sel_one, sel_others[n, :])[0, 1]
#     kin_pair_cor0[m*99:(m+1)*99, :] = kin_pair_cor1
#
# np.mean(kin_pair_cor0)
#
# # Distance
#
# np.linalg.norm(kin_pair.iloc[0, :] -kin_pair.iloc[1, :])
#
# # distance mean compare
# # 100 kinships and 100 nonkinships
#
# kin_pair_rdm = np.random.randint(0, kin_pair.shape[0], 100)
# sel_csv = train_relat.iloc[kin_pair_rdm, :]
#
# kin_diff = np.zeros([100, 1])
# train_relat.iloc[kin_pair_rdm, :]
# for i in range(100):
#     pair1 = sel_csv.iloc[i, 0]
#     pair2 = sel_csv.iloc[i, 1]
#     loc1 = np.where(np.asarray(list(faces.keys())) == pair1)[0]
#     loc2 = np.where(np.asarray(list(faces.keys())) == pair2)[0]
#     diff = np.linalg.norm(kin_pair.iloc[loc1[0], :] - kin_pair.iloc[loc2[0], :])
#     kin_diff[i, :] = diff
#
# np.mean(kin_diff)
#
#
# nonkin_pair_rdm = np.random.randint(0, kin_pair.shape[0], 200)
# nonkin_diff = np.zeros([100, 1])
# for i in range(100):
#     pind1 = nonkin_pair_rdm[i]
#     pair1 = kin_pair.iloc[pind1, :]
#     pind2 = nonkin_pair_rdm[i+1]
#     pair2 = kin_pair.iloc[pind2, :]
#     non_diff = np.linalg.norm(pair1 - pair2)
#     nonkin_diff[i, :] = non_diff
#
# np.mean(nonkin_diff)
#
# kin_perct = np.zeros([100, 1])
# for i in range(100):
#     kin1  = kin_diff[i, :]
#     cnt = 0
#     for j in range(100):
#         if nonkin_diff[j, :] < kin1:
#             cnt +=1
#     pert = cnt/100
#     kin_perct[i, :] = pert
#
# np.mean(kin_perct)


os.chdir('/Users/Michael/Documents/MachineLearning/Project')

# Load the dataset
data_frame = pd.read_csv('data_mat.csv', header=None)
data_frame.shape
data_mat = np.asmatrix(data_frame)

# test the data
trainX = data_frame.iloc[:, 0:200]
trainY = data_frame.iloc[:, -1]

# select 1000

# End of Code
