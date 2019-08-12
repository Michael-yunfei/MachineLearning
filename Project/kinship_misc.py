# kinship misc

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


# LDA first


# Function to fit LDA
def fitLDA(X, Y):
    '''
    X and Y are dataframe
    need to be transfomed into:
    Input: X - n by m matrix, containing all features
           Y - n by 1 matrix, contanning the information of class
    Causion: the orders of rows of X and Y are assumed to be corresponding to
             each other, which means inforamtion of any row (say row 10) of X
             and information of same row (row 10) of Y are from same
             observation
    Output: prior_probability: estimated probability: pi_k (dataframe)
            classmean: estiamted mean of different classes: mu_k (dataframe)
            sigma: estimated covariance matrix (matrix)
    '''
    n = X.shape[0]  # get the number of total observation
    columnsIndex = X.columns
    m = X.shape[1]  # get the number of covariate variables (features)
    X = np.asmatrix(np.ascontiguousarray(X)).reshape(n, m)
    Y = np.asarray(np.ascontiguousarray(Y)).reshape(-1, 1)  # Y has to be an array, not matrix
    yunique = np.unique(Y)
    prior_probability = {}  # initialize a distionary for prior probability
    mu_k = {}
    sigma = {}

    for i in yunique:
        rowindex = np.where(Y == i)[0]
        classlabel = 'class' + str(i)
        prior_probability[classlabel] = len(rowindex)/n
        filteredx = X[rowindex, :]
        mean_vector = np.mean(filteredx, axis=0)
        mu_k[classlabel] = mean_vector
        mean_maxtrix = np.repeat(mean_vector, filteredx.shape[0], axis=0)
        diff_matrix = filteredx - mean_maxtrix
        sigma_k = diff_matrix.T @ diff_matrix
        # calcluate within-class covariance
        sigma[classlabel] = sigma_k

    # tidy the output
    sigma_sum = np.zeros([m, m])
    for i in sigma:
        sigma_sum += sigma[i]
    sigma = sigma_sum/(n - len(yunique))  # estimate final sigma
    prior_probability = pd.DataFrame(list(prior_probability.values()),
                                     index=prior_probability.keys(),
                                     columns=['Prior Probability'])
    mean_dataframe = []
    for v in mu_k:
        mean_dataframe.extend(np.array(mu_k[v]))
    mu_k = pd.DataFrame(mean_dataframe, index=mu_k.keys(),
                        columns=columnsIndex)

    return(prior_probability, mu_k, sigma)


# Function to classify LDA
def classifyLDA(featureX, priorpro, mu, sigma, critical=False):
    '''
    This is the classification for multi-categories case
    and it only takes the binary case as a special one

    Input: 1)featureX: n by m dataframe, where n is sample size, m is number
           of covariatle variables

           2)mu: k by m dataframe, where k is the number of classes or
           categories m is the number of covariatle variables. mu is
           taken from fitLDA() function.

           3) priorpro: k by 1 dataframe, prior probabilty, it is taken from
           fitLDA() function.

           4) sigma: k by k covariance matrix, it is also taken from fitLDA()

           5) critical=Faulse, if it is true, then it should be the case that
              number of calsses = number of features. Otherwise, there is
              no solution for critical values.
              WARNING: in this function, the critical value calculation only
                       applies for the binar case with one feature
    Output:
           Classification results: n by 1 vector and newdataframe with
                                   extra column called 'LDAClassification'
           and k by 1 vector of critical values of X
    '''
    newX = pd.DataFrame.copy(featureX)
    classLabels = priorpro.index  # get class labes from dataframe
    featureLabels = featureX.columns
    meanLabels = mu.columns
    X = np.asmatrix(np.ascontiguousarray(featureX))
    priorpro = np.asmatrix(priorpro)
    if all(featureLabels == meanLabels):
        delta = np.zeros([featureX.shape[0], 1])
        for v in range(len(classLabels)):
            Probabilty = np.array(priorpro[v, :]).reshape(-1, 1)
            # get prior probabilty for class k
            mean_vector = np.array(mu.iloc[v, :]).reshape(-1, 1)
            # get mean vector for class k
            deltaX = (X @ np.linalg.inv(sigma) @ mean_vector
                      - 1/2 * mean_vector.T @
                      np.linalg.inv(sigma) @ mean_vector
                      + math.log(Probabilty))
            delta = np.hstack([delta, np.asmatrix(deltaX).reshape(-1, 1)])

        delta = delta[:, 1:]
        classificationResults = np.argmax(delta, axis=1)
        # maximize the delta over k
        newX['LDAClassification'] = classificationResults.reshape(-1, 1)
    else:
        print('Pleasre make sure that featured X and mean vector\
              have the same covariate variables')

    if critical is True:
        if len(classLabels) < len(featureLabels):
            print('There is no solutions for critical values\
                  as dimension of classes is less than dimension\
                  of covariate variables')
        else:
            # calculate the critical values
            mean_i = np.array(mu.iloc[0, :]).reshape(-1, 1)
            mean_j = np.array(mu.iloc[1, :]).reshape(-1, 1)
            prob_i = np.array(priorpro[0, :]).reshape(-1, 1)
            prob_j = np.array(priorpro[1, :]).reshape(-1, 1)
            xcritical = sigma/(mean_j - mean_i)*(
                math.log(prob_i/prob_j) + (mean_j**2
                                           - mean_i**2)/(2*sigma))
            return(classificationResults, newX, xcritical)
    else:
        return(classificationResults, newX)


# Function to compute 0-1 loss and type I/II error for classification
def computeLoss(y, yhat, binary=False):
    '''
    Input: true classification y (dataframe n by 1)
           estimated classifcaiton yhat (dataframe n by 1)
           binary=False: if it is faulse, only zeroone loss is returned
                         if it is true, the class of y has to be binary,
                         then type I/II error table is returned
    Output:
          0-1 loss (dataframe) if binary=False
          type I/II error (dataframe) if binary=True
    '''
    n = y.shape[0]
    y = np.array(y).reshape(-1, 1)
    yhat = np.array(yhat).reshape(-1, 1)
    if binary is True:
        yunique = np.unique(y)
        y1Index = np.where(y == yunique[1])
        predict_positiveRatio = np.sum(yhat[y1Index] == yunique[1])/len(y1Index[0])
        type2error = 1 - predict_positiveRatio
        y0Index = np.where(y == yunique[0])
        predict_negativeRatio = np.sum(yhat[y0Index] == yunique[0])/len(y0Index[0])
        type1error = 1 - predict_negativeRatio
        typeError = pd.DataFrame(np.array([[predict_positiveRatio,
                                            type2error],
                                           [predict_negativeRatio,
                                            type1error]]),
                                 columns=['Predicted positive',
                                          'Predicted negative'],
                                 index=['class'+str(yunique[1]),
                                        'class'+str(yunique[0])])
        return(typeError)
    else:
        zeroOneloss = 1 - (sum(y == yhat)/n)
        return(zeroOneloss)


# begin to train the dataset
LDA_trainX = pd.DataFrame(kin_dataset[:, :400])
LDA_trainY = pd.DataFrame(kin_dataset[:, 400:401])
binary_pro, binary_mean, binary_sigma = fitLDA(LDA_trainX, LDA_trainY)

LDA_preictedY, LDA_newX = classifyLDA(LDA_trainX, binary_pro,
                                      binary_mean, binary_sigma,
                                      critical=False)


loss = computeLoss(LDA_trainY, LDA_preictedY)
print(loss)  # [0.1682243]
typeloss = computeLoss(LDA_trainY, LDA_preictedY, True)
print(typeloss)


# QDA


def fitQDA(X, Y):
    '''
    X and Y are dataframe
    need to be transfomed into:
    Input: X - n by m matrix, containing all features
           Y - n by 1 matrix, contanning the information of class
    Causion: the orders of rows of X and Y are assumed to be corresponding to
             each other, which means inforamtion of any row (say row 10) of X
             and information of same row (row 10) of Y are from same
             observation
    Output: prior_probability: estimated probability: pi_k (dataframe)
            classmean: estiamted mean of different classes: mu_k (dataframe)
            sigma: estimated covariance matrix (matrix)
    '''
    n = X.shape[0]  # get the number of total observation
    columnsIndex = X.columns
    m = X.shape[1]  # get the number of covariate variables (features)
    X = np.asmatrix(np.ascontiguousarray(X)).reshape(n, m)
    Y = np.asarray(np.ascontiguousarray(Y)).reshape(-1, 1)  # Y has to be an array, not matrix
    yunique = np.unique(Y)
    prior_probability = {}  # initialize a distionary for prior probability
    mu_k = {}
    sigma = {}

    for i in yunique:
        rowindex = np.where(Y == i)[0]
        classlabel = 'class' + str(i)
        prior_probability[classlabel] = len(rowindex)/n
        filteredx = X[rowindex, :]
        mean_vector = np.mean(filteredx, axis=0)
        mu_k[classlabel] = mean_vector
        mean_maxtrix = np.repeat(mean_vector, filteredx.shape[0], axis=0)
        diff_matrix = filteredx - mean_maxtrix
        sigma_k = (diff_matrix.T @ diff_matrix)/(n-1)
        # calcluate within-class covariance
        sigma[classlabel] = sigma_k

    # tidy the output
    prior_probability = pd.DataFrame(list(prior_probability.values()),
                                     index=prior_probability.keys(),
                                     columns=['Prior Probability'])
    mean_dataframe = []
    for v in mu_k:
        mean_dataframe.extend(np.array(mu_k[v]))
    mu_k = pd.DataFrame(mean_dataframe, index=mu_k.keys(),
                        columns=columnsIndex)

    return(prior_probability, mu_k, sigma)


# Define QDA calssification
def classifyQDA(featureX, priorpro, mu, sigma):
    '''
    This is the classification for multi-categories case
    and it only takes the binary case as a special one

    Input: 1)featureX: n by m dataframe, where n is sample size, m is number
           of covariatle variables

           2)mu: k by m dataframe, where k is the number of classes or
           categories m is the number of covariatle variables. mu is
           taken from fitLDA() function.

           3) priorpro: k by 1 dataframe, prior probabilty, it is taken from
           fitLDA() function.

           4) sigma: k by 1 dictionary, each erray is a m by m matrix
    Output:
           Classification results: n by 1 vector and newdataframe with
                                   extra column called 'LDAClassification'
           and k by 1 vector of critical values of X
    '''
    newX = pd.DataFrame.copy(featureX)
    classLabels = priorpro.index  # get class labes from dataframe
    featureLabels = featureX.columns
    meanLabels = mu.columns
    X = np.asmatrix(np.ascontiguousarray(featureX))
    if all(featureLabels == meanLabels):
        delta = np.zeros([featureX.shape[0], 1])
        deltaX = np.zeros([featureX.shape[0], 1])
        for v in classLabels:
            probabilty = np.array(priorpro.loc[v, :])
            mean_vector = np.array(mu.loc[v, :]).reshape(-1, 1)
            sigma_k = sigma[v]
            for i in range(featureX.shape[0]):
                    x_rowvector = X[i, :]
                    deltax = (np.log(probabilty)
                              -0.5*np.log(np.linalg.det(sigma_k))
                              -0.5*x_rowvector@np.linalg.inv(sigma_k)@x_rowvector.T
                              +x_rowvector@np.linalg.inv(sigma_k)@mean_vector
                              -0.5*mean_vector.T@np.linalg.inv(sigma_k)@mean_vector)
                    delta[i, :] = deltax
            deltaX = np.hstack([deltaX, delta])

        deltaX = deltaX[:, 1:]
        classificationResults = np.argmax(deltaX, axis=1)
        # maximize the delta over k
        newX['LDAClassification'] = classificationResults.reshape(-1, 1)
    else:
        print('Pleasre make sure that featured X and mean vector\
              have the same covariate variables')
        sys.exist(0)

    return(classificationResults, newX)


q_pro, q_mean, q_sigma = fitQDA(LDA_trainX, LDA_trainY)

q_preictedY, q_newX = classifyQDA(LDA_trainX, q_pro, q_mean, q_sigma)


Qloss = computeLoss(LDA_trainY, q_preictedY)
print(Qloss)  # [0.17757009]
Qtypeloss = computeLoss(LDA_trainY, q_preictedY, True)
print(Qtypeloss)
