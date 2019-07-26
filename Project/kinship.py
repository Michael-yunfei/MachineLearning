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
train_relat.head()
# p1	p2
# 0	F0002/MID1	F0002/MID3
# 1	F0002/MID2	F0002/MID3
# 2	F0005/MID1	F0005/MID2
# 3	F0005/MID3	F0005/MID2
# 4	F0009/MID1	F0009/MID4

# where p1 and p2 indicate the relationships
f1 = mtmg.imread('./train/F0002/MID1/P00009_face3.jpg')
f2 = mtmg.imread('./train/F0002/MID2/P00009_face2.jpg')
f3 = mtmg.imread('./train/F0002/MID3/P00009_face1.jpg')
f5 = mtmg.imread('./train/F0005/MID1/P00053_face1.jpg')
f6 = mtmg.imread('./train/F0016/MID1/P00162_face1.jpg')
plt.imshow(f1)
plt.imshow(f2)
plt.imshow(f3)

# in this example, f1-f3 are kin (father-daughter);
# f2-f3 are kin (mother-daughter);
# no realtionship between f5 with f1 to f3;

f1.shape  # (224, 224, 3)
f5.shape  # (224, 224, 3)

# we got 3 dimensions of data set

f1[:, :, 0]
f1[:, :, 1]
f1[:, :, 2]

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
# their eyes structure are very kin

# filter out the third dimension
f1hv = f1[:, :, 2]
f2hv = f2[:, :, 2]
f3hv = f3[:, :, 2]
f5hv = f5[:, :, 2]
f6hv = f6[:, :, 2]
f2hv.shape
plt.imshow(f2hv)
plt.imshow(f3hv)

# we will work on the first two dimensions (fiter condition [:, :, 3])
# check the corros correlation
f2_f3_corr = signal.correlate2d(f2hv, f3hv)

plt.imshow(f2_f3_corr)
np.linalg.norm(f2_f3_corr)  # 65898.89399678874

f3_f5_corr = signal.correlate2d(f3hv, f5hv)
np.linalg.norm(f3_f5_corr)  # 65887.66007379531

f3_f6_corr = signal.correlate2d(f3hv, f6hv)
np.linalg.norm(f3_f6_corr)  # 65840.48157478802

# it seams that simple correlation can recognize the kinship


# define the standarize the mattrix
def matstd(A):
    B = (A - np.mean(A))/np.std(A)
    return B


# standardlize the amtrix
f1hvsd = matstd(f1hv)
f2hvsd = matstd(f2hv)
f3hvsd = matstd(f3hv)
f5hvsd = matstd(f5hv)
f6hvsd = matstd(f6hv)


cor23 = signal.correlate2d(f2hvsd, f3hvsd)
np.linalg.norm(cor23)  # 2014037.863989161 -- mother and daughter

cor13 = signal.correlate2d(f1hvsd, f3hvsd)
np.linalg.norm(cor13)  # 1651752.7813143819 -- father and daughter

cor15 = signal.correlate2d(f1hvsd, f5hvsd)
np.linalg.norm(cor15)  # 1503840.8501614681 -- father with stranger

cor16 = signal.correlate2d(f1hvsd, f6hvsd)
np.linalg.norm(cor16)  # 1232214.3310698608 -- father with male stranger
plt.imshow(f6)

cor25 = signal.correlate2d(f2hvsd, f5hvsd)
np.linalg.norm(cor25)  # 1989398.879514042 -- mother with female stranger

cor35 = signal.correlate2d(f3hvsd, f5hvsd)
np.linalg.norm(cor35)  # 1738723.8026003127 -- daughter with female stranger

cor26 = signal.correlate2d(f2hvsd, f6hvsd)
np.linalg.norm(cor26)  # 1639486.6729691173 -- mother with male stranger

cor36 = signal.correlate2d(f3hvsd, f6hvsd)
np.linalg.norm(cor36)  # 1420378.095979528 -- daughter with male stranger

# after standardize the matrix, the difference is more striking
# but we need know the genders as the correlation with genders

###############################################################################
# Section 2 - access 200 picutres and calculate the cross correlations
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


path = '/Users/Michael/Documents/MachineLearning/Project/train/'
faces = {}  # initialize the dictionary to store the data
# read 100 pairs from train_relationships.csv
for i in range(100):
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
len(faces.keys())  # 58, unique values
faces['F0002/MID1'][:, :, 2]

# Now, we transfer the matrix into 2 dimension
for m, n in enumerate(faces):
    faces[n] = faces[n][:, :, 2]
    plt.imshow(faces[n])

# %matplotlib qt
# %matplotlib inline

faces_gender = {}  # initialize the gender dictionaries
# Now, we classify them as male and female mannually
for m, n in enumerate(faces):
    faces[n] = faces[n]
    plt.imshow(faces[n])
    plt.show()
    gender = input("Tell me the gender of the picture shown to you")
    faces_gender[n] = gender


# randomly check some gender tags
filenames = list(faces.keys())
filenames[5]
print(len(filenames))

randindx = np.random.randint(0, 57, 10)
for i in randindx:
    plt.imshow(faces[filenames[i]])
    plt.show()
    print(faces_gender[filenames[i]])

# once we have genders tagged on each picture
# we can calcuate their cross correlations based on gender differences

# Standardize the data first
# define the standarize the mattrix


def matstd(A):
    B = (A - np.mean(A))/np.std(A)
    return B


faces_std = {}  # initialize another dictionary
for m, n in enumerate(faces):
    faces_std[n] = matstd(faces[n])

# initialize a dataframe to store all realtionships
croxcor_mat = pd.DataFrame(np.zeros([100, 6]))
for i in range(100):
    face1 = train_relat.iloc[i, 0]  # get pairs' name
    face2 = train_relat.iloc[i, 1]
    croxcor_mat.iloc[i, 0] = face1
    croxcor_mat.iloc[i, 1] = faces_gender[face1]
    croxcor_mat.iloc[i, 2] = face2
    croxcor_mat.iloc[i, 3] = faces_gender[face2]
    croxcor = signal.correlate2d(faces_std[face1], faces_std[face2])
    croxcor_mat.iloc[i, 4] = np.linalg.norm(croxcor)
    if croxcor_mat.iloc[i, 1] == croxcor_mat.iloc[i, 3]:
        croxcor_mat.iloc[i, 5] = 'S'
    else:
        croxcor_mat.iloc[i, 5] = 'D'

croxcor_mat.to_csv("output.csv", index=False)

# once we have this cross correlation dataframe, we can have some meaningful
# exploration right now
croxcor_mat.head()
# 0	1	2	3	4	5
# 0	F0002/MID1	m	F0002/MID3	f	1.335473e+06	D
# 1	F0002/MID2	f	F0002/MID3	f	2.036861e+06	S
# 2	F0005/MID1	f	F0005/MID2	m	1.985822e+06	D
# 3	F0005/MID3	m	F0005/MID2	m	1.697405e+06	S
# 4	F0009/MID1	m	F0009/MID4	m	1.335648e+06	S

# we select one picture F0002/MID1, and we randomly select 10 femals
# to calcuate the cross correlation
# if it is larger than the kinship value, then we should study that picture
ict = 0
for m, n in enumerate(faces_gender):
    if faces_gender[n] == 'f':
        femal_face = faces_std[n]
        cor_test = signal.correlate2d(faces_std['F0002/MID1'], femal_face)
        cor_norm = np.linalg.norm(cor_test)
        if cor_norm > croxcor_mat.iloc[0, 4]:
            print('You got one')
            print(cor_norm)
            print(n)
            plt.imshow(femal_face)
            plt.show()
        ict += 1
    if ict == 10:
        break


# check female
ict = 0
for m, n in enumerate(faces_gender):
    if faces_gender[n] == 'f':
        femal_face = faces_std[n]
        cor_test = signal.correlate2d(faces_std['F0002/MID3'], femal_face)
        cor_norm = np.linalg.norm(cor_test)
        if cor_norm > croxcor_mat.iloc[1, 4]:
            print('You got one')
            print(cor_norm)
            print(n)
            plt.imshow(femal_face)
            plt.show()
        ict += 1
    if ict == 10:
        break

# it looks like this ideas do not work, so we change ideas

# transfer the matrix into the vector format
faces['F0002/MID3'].shape
plt.imshow(faces['F0002/MID3'][1:120, :])

faces['F0002/MID3'].reshape(1, -1)

# initialize the matrix
faces_mat = np.zeros([len(list(faces.keys())), 224*224])
for m, n in enumerate(faces):
    face_vector = faces[n].reshape(1, -1)
    faces_mat[m, :] = face_vector


faces_mat.shape  # (58, 50176)

# creat gender labels 0 - female and 1 - male
gender_labels = np.zeros([len(list(faces.keys())), 1])
for m, n in enumerate(faces_gender):
    if faces_gender[n] == 'f':
        gender_labels[m, 0] = 0
    else:
        gender_labels[m, 0] = 1

gender_labels

ohe = OneHotEncoder(sparse=False)
y_mat = ohe.fit_transform(gender_labels)
y_mat.shape  # (58, 2)

y_mat

# try network classify for genders


























# End of Code
