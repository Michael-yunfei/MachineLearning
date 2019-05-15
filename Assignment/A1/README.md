
<h1 align="center">
  <a>Assignment 1.A – Linear Regression</a>
</h1>
<h3 align="center">
  <a>Machine Learning @ Uni-Konstanz 2018</a>
</h3>

## Idea 📓

- Code a simple linear (multivariate) regression model.

- Train the linear model (estimate parameters) on a given training dataset.

- Test the trained model on a given testing dataset.

- Plot the results for training and testing stages.

## Data 📦

- [M&Ms dataset](https://gist.github.com/giob1994/ffcd8c72a8a5477219aca9c5884c2094): a small dataset about the number of colored M&Ms in packets and the packets weight. The dataset is already provided in `MandMs.mat`.

## Tasks 📝

#### Preliminaries

- Load the dataset from `MandMs.mat` determine the names of the variables in the dataset with the command `who`.

- Write a function 

	```matlab
	[trainSample, testSample] = splitSample(sample, trainSize, permute) 
	```
	
	that splits the sample into a *training sample* and a *test sample*. 
	
	- `trainSize` is a number between 0 and 1 that decides which percentage of the sample is to be used *for training*.
		
	- `permute` is a boolean value such that if `true` then the sample is *randomly permuted* before being split, if `false` then no permutation is used.
	
#### Regression

***

_In the following tasks you are asked to use *directly* the **normal equations**_ (see. ***Lecture 2***, slides 45-46) _to carry out some simple regression tasks on the M&Ms dataset._

_You will also need to code the l1, l2 (MSE) and Huber losses_ (see. ***Lecture 2***, slides 17-18, 42). _For the [Huber loss](https://en.wikipedia.org/wiki/Huber_loss), consider δ=1 and write a function `lossHuber(X, Y)` that computes the mean Huber loss over the sample_.

***

1. Construct a *linear regression model* that predicts the weight of a given bag provided the number of **red** candies in a bag. 

  	- Use 80% of the sample length for training and 20% for testing.
  	
  	- Compute the training (in-sample) and testing (out-of-sample) performance of such model using l1, l2 (MSE) and Huber losses.

  	- Plots:
  	
  		- Predicted weight values (regression line), predicted data points and ground-truth data points for the **training data**.
  	
  		- Predicted weight values (regression line) and ground-truth data points for the **test data**.


2. Construct an example of **multivariate regression** choosing the colors **green** and **blue**.

  	- Compute the training (in-sample) and testing (out-of-sample) performance of such model using l1, l2 (MSE) and Huber losses.

  	- Plot *in 3D* the predicted weight values (regression plane) and the actual ones.
  
3. Conclude which model is performing **better** for the **last 20%** _in the dataset chosen as test data_.

## Notes ⚠️

**Write your assignment code in the `main.m` file**: this is the script that has to contain the *core* of your assignment. 

Please *avoid creating unnecessary scripts/function files*, as this makes the code harder to grasp in its entirety.

**Good programming rules apply**:
- Use meaningful variable names. 
- Use indentation.
- Keep your code tidy. 
- Add a minimum of comments (if you deem then necessary). 

<br>

***Good work!***
