
<h1 align="center">
  <a>Assignment 4 – Discriminative Classifiers- due to 02.07.19</a>
</h1>
<h3 align="center">
  <a>Machine Learning @ Uni-Konstanz 2019</a>
</h3>

## Idea 📓

- Train and test the Logistic classifier, Perceptron, and k nearest neighbors (kNN).

- Binary and multi-class classification.

- Generalizing classifiers to d>=2 features.

***

**The goal of all these tasks is to learn how to construct and train the Logistic classifier, Perceptron, and k nearest neighbors (kNN).** You will have to carry out both binary and 3-class classifications, depending on the type of classifier at hand.

***

## Data 📦

- [Wine Dataset](http://archive.ics.uci.edu/ml/datasets/Wine):
	
	> These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines. 
	
	You will find the the dataset in the `wine.csv` file:
		
	- Column **1**: classes.
		
	- Columns **2-14**: features.

## Tasks 📝

#### Preliminaries

- Before starting with the main code, you will need to create the following functions:
	
	- `fitLogistic.m`
	
    - `classifyLogistic.m`
    
    - `fitPerceptron.m`
    
    - `classifyPerceptron.m`
    
    - `classifyKNN.m`

Each function file contains the necessary instructions and starter code.

***

**You can re-use any of the functions of Assignment 3, if needed.**

***

	
#### Classification

The file `main.m` will be divided as follows:
	
1. **Part I: Load Data**.

	- Load the `wine.csv` dataset.

2. **Part II: Binary/Multiclass Classification with Two/Multiple Features**:

	- *Task 1* : Construct the logistic classifier
	
	- *Task 2* : Construct the Perceptron classifier for binary classification.
	
	- *Task 3* : Construct the KNN classifier for binary classification.
	
	
  
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
