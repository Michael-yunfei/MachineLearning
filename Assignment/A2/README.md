
<h1 align="center">
  <a>Assignment 2 – Linear and Polynomial Regression with Multiple Variables</a>
</h1>
<h3 align="center">
  <a>Machine Learning @ Uni-Konstanz 2019- Due  to 29.05.2019 23:59</a>
</h3>

## Idea 📓

- Linear and polynomial regression using multiple regressors.

- Use Normal Equations (OLS-Regression), Batch Gradient Descent, and Stochastic Gradient Descent to fit the models.

- Get acquainted with phenomena of under- and overfitting.

***

**The goal of all these tasks is to illustrate that one can propose a model for which the training error will be as close to zero as one wants. 
The goal, however, is to construct a model with good performance at the training phase but at the same time with good generalization properties during the testing phase.** In the following exercises you will learn the model selection techniques and will apply them to some of the models in this assignment.

***

## Data 📦

- [World Happiness Report](https://www.kaggle.com/unsdsn/world-happiness):
	
	> The happiness scores and rankings use data from the Gallup World Poll. The scores are based on answers to the main life evaluation questions asked in the poll.
	
	You will find the *2016* version of the dataset.

## Tasks 📝

#### Preliminaries

- Before starting with the main code, you will need to complete the following functions:
	
	- `computeLossLinModel.m`
	
	- `computeLoss.m`
	
	- `featureNormalize.m`
	
	- `gradientDescent.m`
	
	- `gradientDescentStochastic.m`
	
	- `normalEqn.m`

Each function file contains the necessary instructions and the starter code.

***

*If you need a refresher on Gradient Descent and its implementation, take a look at* [Hands-on Machine Learning with Scikit-Learn and Tensorflow]
(https://konstanz.summon.serialssolutions.com/#!/search?bookMark=ePnHCXMwVV3NCsIwDJ7owR98h_oABXXptngVxy7iZffRrYmIMsHt_TGZQ9itKSVQmpJ8afJ1HW291ma3_dDDFWYjrRLKUR8lvp9PZMTlPyGC4MTiVtGp0CZXe2vNdagnJDNSjd6N5iaNGP3z0dth0shSUwrge3_yl5I6LlpBbpuozC_lubDjfwK2AwTbMJHcJiWIh1rGmKQBQ0zks0AO2AHU-9Qn4gIJwyFjJnGHkDExpMHr0_Tup7bzLEC20jC4qya7i7_5C0d2). *Chapter 4 contains the key formulas that will get you started with GD.*

***

	
#### Regression

The file `main.m` will be divided as follows:
	
1. **Part I: Load Data**.

2. **Part II: Linear Regression with One Regressor**:

	- *Task 1* : Create the regressor (input, feature) X using the "Freedom" score and create the explained variable (output, response) Y from "HappinessScore".
	
	- *Task 2* : Train a linear regression model (see Assignment 1) using the whole sample and the created regressor X. Use the **normal equations (OLS/closed form solution)**.
	
	- *Task 3* : Check the computations with built-in regress function (depending on the system of implementation).
	
	- *Task 4* : Train a linear regression model (see *Task 2*) using **batch gradient descent**.
	
	- *Task 5* : Train a linear regression model (see *Task 2*) using **stochastic (online) gradient descent**. Train a linear regression model using **stochastic (online) gradient descent** but instead of using the whole sample,
		     use a 5-fold crossvalidation. Compare the results.
		     
3. **Part III: Polynomial Regression with One Regressor**:

	- *Task 6* : Construct k = 10 polynomial regression models (nonlinear models). Use the **normal equations (OLS/closed form solution)**.
	-  *Task 7* : Redo *Task 6*, but with **stochastic (online) gradient descent**.
	
4. **Part IV: Regression with Two Regressors**:
	
	- *Task 8* : Use the "Freedom" score and the "Family" score as regressors and again let the explained variable (output, response) Y be "HappinessScore". Train the linear model with the help of the **batch gradient descent**.
	
	- *Task 9* : Take a regression function as a degree-2 polynomial in two ariables: freedom and family. Estimate the parameters with **normal equations**.

## Notes ⚠️

**Write your assignment code following the detailed instructions given in  `main.m`**.

Please *avoid creating unnecessary scripts/function files*, as this makes the code harder to grasp in its entirety.
Feel free to do different task in one section if you safe time by doing so. But, make sure your code is still well structured.

**Good programming rules apply**:
- Use meaningful variable names and label your plots.  Use indentation.
- Keep your code tidy. 
- Add a minimum of comments (if you deem then necessary). 

<br>

***Good work!***
