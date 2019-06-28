%%%%%%%%%%%%%  DUE TO 02.07.2019 23:59 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%  Discriminative classifiers for binary and multiclass classification with 
%  multiple features

% You can continue working with the files from the Assignment 3
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you to get started on the
%  generative classification exercise. 
%
%  You will need to create the following functions in this 
%  exericse:
%
%     classifyKNN.m
%     fitLogistic.m
%     classifyLogistic.m
%     fitPerceptron.m
%     classifyPerceptron.m
%     
%
%  Data description (see http://archive.ics.uci.edu/ml/datasets/Wine)
%  ------------------------------------------------------------------
%
%  > These data are the results of a chemical analysis of
%    wines grown in the same region in Italy but derived from three
%    different cultivars.
%    The analysis determined the quantities of 13 constituents
%    found in each of the three types of wines. 
%
%    Number of Instances
% 
%         class 1 -> 59
% 	      class 2 -> 71
% 	      class 3 -> 48
% 
%    Number of Attributes 
% 	
%         13
%     
%% Task 0 : Data Setup 
%  Clear and Close Figures
clear; close all; clc

fprintf('Loading data ...\n');

%  Load the 'wine.csv' dataset and determine how many classes there are in
%  the dataset. Create separate variables containing the class labels and all the 
%  available features. 


dataset = csvread('wine.csv');
fprintf('Data successfully loaded.\n')
classes = dataset(:, 1);
num_classes = unique(classes);

indOfClass1 = logical(classes == 1);
indOfClass2 = logical(classes == 2);
indOfClass3 = logical(classes == 3);

X_all = dataset(:, 2:end);
descrX = {'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', ...
    'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',...
    'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',...
    'Proline'};
n1 = sum(indOfClass1);
n2 = sum(indOfClass2);
n3 = sum(indOfClass3);

n = length(classes);
if n ~= n1 + n2 + n3
    fprintf('something wrong in the computation of class representatives');
end


% Run  all  classification on the following data

% -------------------------------------------------------------------------
% Wine 1/Wine 3 with the features
%   - 'Proanthocyanins'
%   - 'Alcalinity of ash'
% -------------------------------------------------------------------------

indOfClasses = indOfClass1 | indOfClass3;
indOfFeature_1 = find(strcmp(descrX, 'Proanthocyanins'));
indOfFeature_2 = find(strcmp(descrX, 'Alcalinity of ash'));
d = 2;
featuresSet = dataset(indOfClasses, 1 + [indOfFeature_1, indOfFeature_2]);

Y = dataset(indOfClasses, 1);

Y(Y == 1) = 1;
Y(Y == 3) = 0;

classLabels = [1,0];
features_Class_1 = featuresSet(Y == 1, :);
features_Class_3 = featuresSet(Y == 0, :);

% undersampling
[n1, ~] = size(features_Class_1);
[n2, ~] = size(features_Class_3);

minlength = min(n1, n2);
features_Class_1 = features_Class_1(1:minlength, :);
features_Class_3 = features_Class_3(1:minlength, :);

trainSize = 0.9;
rng('default') % for reproducibility

permute = 1;

[trainClass1, testClass1] = splitSample(features_Class_1, trainSize, permute);
[trainClass3, testClass3] = splitSample(features_Class_3, trainSize, permute);

[train_length, ~] = size(trainClass1);

train_Y = [ones(train_length, 1); zeros(train_length, 1)];
train_featuresSet = [trainClass1; trainClass3];

[train_featuresSet_norm, mu_featuresSet, sigma_featuresSet] = featureNormalize(train_featuresSet);

[test_length, ~] = size(testClass1);

test_Y = [ones(test_length, 1); zeros(test_length, 1)];
test_featuresSet = [testClass1; testClass3];

[test_featuresSet_norm] = featureNormalize(test_featuresSet, mu_featuresSet, sigma_featuresSet);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Task 1 : Construct the logistic classifier. For that fill in the function
% fitLogistic and classifyLogistic using the same approach as in Assignment
% 3. Both functions should be constructed in order to
% work with multiple classes and multiple features if needed. 
% Use regularized logistic classifier (with 
% l1 or l2 regularization option).Use all the functions form the previous
% assignments when relevant (spliting of sample, randomization, losses, 
% stochastic gradient descent)
% Classify the train and test sample with a l2-logistic classifier.
% The value of the regularization parameter needs to be selected with the 5-fold
% cross-validation procedure. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Task 2 : Construct the perceptron classifier for binary classification. 
% For that fill in the function fitPerceptron and classifyPerceptron using 
% the same approach as in Assignment 3. Both functions should be constructed in order to
% work with multiple features if needed. 
% Compute the k (number of updates).
% Use all the functions form the previous
% assignments when relevant (spliting of sample, randomization, losses, 
% stochastic gradient descent)
% Report the testing and training errors using the hinge loss. (Add one
% more option for the function computeLoss from the previous assignments).
% Plot the results of the classification for both sets
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Task 3: Construct the kNN classifier for binary classification. 
% For that fill in the function classifyKNN. 
% Choose the best k parameter using the  10-fold cross-validation (CV) procedure.
% Use all the functions from the previous
% assignments when relevant (spliting of sample, randomization, losses, 
% stochastic gradient descent)
% Classify the test set and report the empirical error using the 0-1 loss and the CV chosen k.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





