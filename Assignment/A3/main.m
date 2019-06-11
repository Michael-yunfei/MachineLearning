%  Generative classifiers for binary and multiclass classification with one
%  and multiple features
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you to get started on the
%  generative classification exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     fitLDA.m
%     classifyLDA.m
%     fitQDA.m
%     classifyQDA.m
%     computeLoss.m
%     fitNaiveBayesGauss.m
%     classifyNaiveBayesGauss.m
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
%% Part I: Load Data ======================================================

%  Clear and Close Figures
clear; close all; clc

fprintf('Loading data ...\n');

%  Load the 'wine.csv' dataset and determine how many classes there are in
%  the dataset. Create separate variables containing the class labels and all the 
%  available features. Create a variable containing the names of the features,
%  for that look at the description of the data following the link provided
%  above. Determine how many representatives of each class
%  there are in the dataset

dataset = csvread('wine.csv');

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

fprintf('Data succesfully loaded. \n')
%% Part II: Binary Classification with One Feature ========================

% Select only classes 1 and 3 for this part and feature 'Proanthocyanins'.
% In this binary classification exercise assign label 0 to Class 1 and
% label 1 to Class 3.

indOfClasses = indOfClass1 | indOfClass3;
indOfFeature = find(strcmp(descrX, 'Proanthocyanins'));

featuresSet = X_all(indOfClasses, indOfFeature);
Y = dataset(indOfClasses, 1);

Y(Y == 1) = 1;
Y(Y == 3) = 0;

classLabels = [1,0];
features_Class_1 = featuresSet(Y==1);
features_Class_3 = featuresSet(Y==0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 0: Plot the data by creating two <count> density-normalized histograms in 
% two different colors
% of your choice; for that use the specific normalization and 'BinWidth' set to 0.3.
% Add the grid.
% Add descriptive legend and title.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


figure('Name', 'Wine Dataset - Wines 1 and 3 Proanthocyanins Content')
title('Wine Dataset - Wine 1 and 3 Proanthocyanins Content')

hold on

% Plot histogram of Wine 1
histogram(features_Class_1, 'Normalization', 'count', ...
    'EdgeColor', 'black', 'FaceColor', 'red', 'BinWidth', 0.3, 'FaceAlpha', .4);
hold on
% Plot histogram of Wine 3
histogram(features_Class_3, 'Normalization', 'count', ...
    'EdgeColor', 'blue', 'FaceColor', 'blue', 'BinWidth', 0.3, 'FaceAlpha', .4);
grid on
hold off


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 1 : Construct LDA classifier. For that fill in the function fitLDA
% and classifyLDA. Both functions should be constructed in order to
% work with multiple classes and multiple feautures if needed. We start
% here however with only two-classes classification which admits the
% explicit critical decision boundary value.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 2 : Compute the empirical value of the error using the 0-1 loss. 
% For that add typeOfLoss '0-1' option to the function computeLoss from the  
% previous assignment. Additionally, this function needs to output the Type I and Type II
% errors (false positive and false negative) which will be filled in only in the case of
% binary classification.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 3 : Plot the resulting classification.
% Create two histograms in two different colors  of your choice: for these, 
% use the option 'Normalization' set to 'pdf', and 'BinWidth' set to 0.3.
% Superimpose the two normal distributions and the Gaussian mixture distribution that you
% obtain with the parameters computed in the 'fitLDA' function.
% Add the grid.
% Add descriptive legend and title.
% Plot the decision boundary (critical value for the given threshold of interest,
% which is set by default to 1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 4 : Construct QDA classifier. For that fill in the function fitQDA
% and classifyQDA. Both functions should be constructed in order to
% work with multiple classes and multiple features if needed. We start
% here however with only two-classes classification first.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 5 : Compute the empirical value of the error using the 0-1 loss. 
% using the function computeLoss together with the Type I and Type II
% errors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 6 : Plot the resulting classification.
% Create two histograms in two different colors  of your choice: for these, 
% use the option 'Normalization' set to 'pdf', and 'BinWidth' set to 0.3.
% Superimpose the two normal distributions and the mixed distribution that you
% obtain as a result from the 'fitLDA' function.
% Add the grid.
% Add descriptive legend and title.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 7 : Construct Naive Bayes Gauss classifier. For that fill in the 
% function fitNaiveBayesGauss and classifyNaiveBayesGauss. 
% Both functions should be constructed in order to work with multiple 
% classes and multiple features if needed. However, we start with only
% two-classes classification.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 8 : Compute the empirical value of the error using the 0-1 loss. 
% using the function computeLoss together with the Type I and Type II
% errors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 9 : Plot the resulting classification.
% Create two histograms in two different colors  of your choice: for these, 
% use the option 'Normalization' set to 'pdf', and 'BinWidth' set to 0.3.
% Superimpose the two normal distributions and the mixed distribution that you
% obtain as a result from the 'fitNaiveBayesGauss' function.
% Add the grid.
% Add descriptive legend and title.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% Part III: Binary Classification with Two Features ======================

% Select only classes 1 and 3 for this part and features:
%
%   - 'Proanthocyanins'
%   - 'Alcalinity of ash'
%
% In this binary classification exercise assign label 0 to Class 1 and
% label 1 to Class 3.


indOfClasses = indOfClass1 | indOfClass3;
indOfFeature_1 = find(strcmp(descrX, 'Proanthocyanins'));
indOfFeature_2 = find(strcmp(descrX, 'Alcalinity of ash'));

featuresSet = dataset(indOfClasses, 1 + [indOfFeature_1, indOfFeature_2]);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 10 : Construct Naive Bayes Gauss classifier. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the empirical value of the error using the 0-1 loss. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the resulting classification.
% Add the grid.
% Add descriptive legend and title.
% Mark misclassified observations.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%% Part IV: Binary Classification with Many Features ======================

% Select only classes 1 and 3 for this part and features:
%
%   - 'Alcohol'
%   - 'Flavanoids'
%   - 'Proanthocyanins'
%   - 'Color intensity'
%
% In this binary classification exercise assign label 0 to Class 1 and
% label 1 to Class 3.


indOfClasses = indOfClass1 | indOfClass3;
indOfFeature_1 = find(strcmp(descrX, 'Alcohol'));
indOfFeature_2 = find(strcmp(descrX, 'Flavanoids'));
indOfFeature_3 = find(strcmp(descrX, 'Proanthocyanins'));
indOfFeature_4 = find(strcmp(descrX, 'Color intensity'));

featuresSet = dataset(indOfClasses, 1+[indOfFeature_1, indOfFeature_2 ...
                                        indOfFeature_3, indOfFeature_4]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 11 : Construct LDA classifier.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the empirical value of the error using the 0-1 loss. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%% Part V: 3-Classes Classification with Many Features ====================

% Select only classes 1 and 3 for this part and features:
%
%   - 'Alcohol'
%   - 'Flavanoids'
%   - 'Proanthocyanins'
%   - 'Color intensity'
%

indOfFeature_1 = find(strcmp(descrX, 'Alcohol'));
indOfFeature_2 = find(strcmp(descrX, 'Flavanoids'));
indOfFeature_3 = find(strcmp(descrX, 'Proanthocyanins'));
indOfFeature_4 = find(strcmp(descrX, 'Color intensity'));

featuresSet_1f = dataset(:, 1+indOfFeature_1);
featuresSet_2f = dataset(:, 1+[indOfFeature_1, indOfFeature_3]);
featuresSet_4f = dataset(:, 1+[indOfFeature_1, indOfFeature_2 ...
                                        indOfFeature_3, indOfFeature_4]);

Y = dataset(:, 1);

classLabels = [1,2,3];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 17 : Construct QDA classifier for the following:
%
%   - 'Alcohol'
%   - 'Alcohol' + 'Proanthocyanins'
%   - All features listed above 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the empirical value of the errors using the 0-1 loss. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


