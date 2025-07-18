# First Principles Linear Regression Build

A project to create and experiment with linear regression from scratch.

## Contains

- Jupyter notebooks recording the exploration of concepts and development of the model
- "my_reg.py", a class that works as a linear regression should

## Overview

This was a more exploratory project enabling me to learn about how linear regression models work statistically and in practise, and about their limitations.

## Usage

The MyLinearReg class is initialised with two arguments, X, y. These are the features and independent variable respectively. Creating an instance will look like this, `model = MyLinearReg(X, y)`. Then, simple methods can be called such as `model.slope()` to get the coefficients, or `y_preds = model.predict(X_test)` to make predictions from the model. Although full completeness is generally desirable, for this proof of concept we only include a few key methods.

## Examples

An example usage, compared with the sklearn model, will be in "EXAMPLE_nb.ipynb" This will include:

1. Get some data, we are going to use advertising data by github user *selva86*
2. Typically, we perform some simple cleaning and preprocessing, but the dataset is already clean
3. Split into training/testing data 80:20
4. Fit both models on the same split
