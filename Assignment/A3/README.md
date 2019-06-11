
<h1 align="center">
  <a>Assignment 3 – Generative Classification - Due to 12.05.2019</a>
</h1>
<h3 align="center">
  <a>Machine Learning @ Uni-Konstanz 2019</a>
</h3>

## Idea 📓

- Train and test LDA, QDA, and GNB classifiers.

- Binary and multi-class classification.

- Generalizing classifiers to N>2 features.

***

**The goal of all these tasks is to learn how to construct and train the LDA, QDA, and GNB classifiers, which are simple but effective tools for solving (some) classification tasks.** You will have to carry out both binary and 3-class classifications, starting from a single feature up to 4 features: in this exercise, you will see how data that is seemingly impossible to precisely separate in few (<3) dimensions may actually be easily and precisely classified by using nothing more than a hyperplane in higher dimensions.

***

## Data 📦

- [Wine Dataset](http://archive.ics.uci.edu/ml/datasets/Wine):
	
	> These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines. 
	
	You will find the the dataset in the `wine.csv` file:
		
	- Column **1**: classes.
		
	- Columns **2-14**: features.

## Tasks 📝

#### Preliminaries

- Before starting with the main code, you will need to complete the following functions:
	
	- `fitLDA.m`
	
    - `classifyLDA.m`
    
    - `fitQDA.m`
    
    - `classifyQDA.m`
    
    - `fitNaiveBayesGauss.m`
    
    - `classifyNaiveBayesGauss.m`
    
    - `computeLoss.m`

Each function file contains the necessary instructions and starter code.

	
#### Classification

The file `main.m` will be divided as follows:
	
1. **Part I: Load Data**.

	- Load the `wine.csv` dataset and select two types (classes) of wine.

2. **Part II: Binary Classification with One Feature**:

	- *Task 0* : To better visualize the nature of the dataset, create a plot of the feature.
	
	- *Task 1* : Construct the 1D LDA classifier.
	
	- *Task 2* : Compute the LDA training/testing errors.
	
	- *Task 3* : Plot the resulting classification.
	
	- *Task 4* : Construct 1D QDA classifier.
	
	- *Task 5* : Compute the QDA training/testing errors.
	
	- *Task 6* : Plot the resulting classification.
	
	- *Task 7* : Construct 1D Gaussian Naive Bayes classifier.
	
	- *Task 8* : Compute the Naive Bayes Gauss training/testing errors.
	
	- *Task 9* : Plot the resulting classification.


3. **Part III: Binary Classification with Two Features**: 
	
	- *Task 10* : 2D Gaussian Naive Bayes classifier, training/testing errors, plot.
	
	
4. **Part IV: Binary Classification with Many Features**: 
	
	- *Task 11* : 4 features LDA classifier, training/testing errors.
	
	
4. **Part V: 3-Classes Classification with Many Features**: 
	
	
	- *Task 12* : QDA classifier with 3 classes, training/testing errors.
	

	
  
## Notes ⚠️

**Write your assignment code following the instructions given in  `main.m`**.

Please *avoid creating unnecessary scripts/function files*, as this makes the code harder to grasp in its entirety.

**Good programming rules apply**:
- Use meaningful variable names. 
- Use indentation.
- Keep your code tidy. 
- Add a minimum of comments (if you deem then necessary). 

<br>

***Good work!***
