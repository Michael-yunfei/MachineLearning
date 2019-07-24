<h1 align="center">
  <a>Assignment 3 – SVM Classification</a>
</h1>
<h3 align="center">
  <a>Machine Learning @ Uni-Konstanz 2019 due to Tuesday 09.07.2019</a>
</h3>

## Idea 📓

- Train and test SVMs with different datasets.

***

We focus on understanding and comparing the performance of the SVM classification methods seen in class. Depending on the dataset different kernels may suggest high performance.
We study the influence of regularization and kernel hyper-parameters on SVM performance for the case of 2D Datasets.

*** 

## Data 📦

- Linearly separable dataset;  

- Checkerboard dataset;

- 'Ripley' dataset;

***

For an example on how to load these datasets, please take a look at `examples\example_datasets.m`.

***

## Tasks 📝

Ask clarified in the tutorial this assignment can only submitted in MATLAB. You can use all the code from the tutorial 6 and optimize SMVs for 2 more datasets.
For more details see `assignment-3-task.pdf`. The toolbox can be found in tutorial 6 aswell.

## Notes ⚠️

**Write your assignment code following the same rules as for the previous assignments and use the assisting code from Tutorial 6**.

Please *avoid creating unnecessary scripts/function files*, as this makes the code harder to grasp in its entirety.

**Good programming rules apply**:
- Use meaningful variable names. 
- Use indentation.
- Keep your code tidy. 
- Add a minimum of comments (if you deem then necessary). 

<br>

***Good work!***

<br>

### Acknowledgement

This code heavily borrows from the [ML_toolbox](https://github.com/epfl-lasa/ML_toolbox) repository: "_a Machine learning toolbox containing algorithms for dimensionality reduction, clustering, classification and regression along with examples and tutorials which accompany the Master level course [Advanced Machine Learning](http://lasa.epfl.ch/teaching/lectures/ML_MSc_Advanced/index.php)  and [Machine Learning Programming](http://edu.epfl.ch/coursebook/fr/machine-learning-programming-MICRO-401) taught at [EPFL](https://www.epfl.ch/) by [Prof. Aude Billard](http://lasa.epfl.ch/people/member.php?SCIPER=115671)_".

The main authors of the toolbox and accompanying tutorials were the TA's from Spring 2016/2017 semesters:  
[Guillaume de Chambrier](http://chambrierg.com/), [Nadia Figueroa](http://lasa.epfl.ch/people/member.php?SCIPER=238387) and [Denys Lamotte](http://lasa.epfl.ch/people/member.php?SCIPER=231543)

#### 3rd Party Software
This toolbox includes 3rd party software:
- [Matlab Toolbox for Dimensionality Reduction](https://lvdmaaten.github.io/drtoolbox/)
- [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
- [Kernel Methods Toolbox](https://github.com/steven2358/kmbox)
- [SparseBayes Software](http://www.miketipping.com/downloads.htm)

You DO NOT need to install these, they are already pre-packaged in this toolbox.
