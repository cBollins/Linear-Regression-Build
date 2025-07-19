# First Principles Linear Regression Build

A project to create and experiment with linear regression from scratch. The project is a walkthrough of my engineering and raw thought processes of both univariate regression statistics, and the multivariate OLS approach.

## Contains

- Jupyter notebooks recording the exploration of concepts and development of the model.
- "my_reg.py", a univariate exploration of the statistics behind linear regression on a single dimensional approach. This .py file is also an expansion of the approach to multiple dimensions.
- "my_reg_OLS.py", the typical ordinary least squares approach used in multivariate linear regression models. This is the build that will be exhibited in "EXAMPLE.nb.ipynb".

## Overview

This project is an exploratory deep dive into the statistical foundation and practical application of linear regressors. 

## Usage

The MyLinearReg class is initialised with two arguments, X, y. These are the features and independent variable respectively. Creating an instance will look like this, `model = MyLinearReg(X, y)`. Then, simple methods can be called such as `model.get_coefficients()` and `model.get_intercept()` to get the parameters, or `y_preds = model.predict(X_test)` to make predictions from the model. Although full completeness is generally desirable, for this proof of concept we only include a few key methods.

## Examples

An example usage, compared with the sklearn model, will be in "EXAMPLE_nb.ipynb" This will include:

1. Get some data, we are going to use advertising data by github user ***selva86***.
2. Typically, we perform some simple cleaning and preprocessing, but the dataset is already clean.
3. Split into training/testing data 80:20.
4. Fit both models on the same split.
5. See if both models came to a similar conclusion. We will see what coefficients and intercepts they both fitted to and the mean-squared error of each.

## Main takeaways

This project grounded my understanding of statistics and how machine learning approaches have been refined around our understanding. Constructing the model from the ground up gave me a grasp not just on the mathematical backbone, but the assumptions, strengths and real-world limitations faced.