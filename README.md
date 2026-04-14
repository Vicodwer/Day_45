# Model Pipeline Notebook

## Overview

This Jupyter Notebook implements a simple end-to-end machine learning
pipeline combining **data auditing**, **data cleaning**, and a **basic
neural network** built from scratch using NumPy.

------------------------------------------------------------------------

## Features

-   Data quality inspection (missing values, unique values, data types)
-   Automated data cleaning
-   Simple feedforward neural network implementation
-   Training loop with loss monitoring
-   Basic evaluation function

------------------------------------------------------------------------

## Notebook Structure

### 1. Data Audit

Analyzes dataset columns and reports: - Missing values - Unique values -
Data types

### 2. Data Cleaning

-   Removes duplicate rows
-   Fills missing numerical values with median
-   Fills missing categorical values with mode

### 3. Neural Network

-   Sigmoid activation function
-   Weight initialization
-   Forward propagation

### 4. Training

-   Mean Squared Error loss
-   Iterative training loop
-   Loss printed every 100 epochs

### 5. Evaluation

-   Simple accuracy calculation

------------------------------------------------------------------------

## Requirements

-   Python 
-   NumPy
-   Pandas
-   Jupyter Notebook

------------------------------------------------------------------------

## How to Use

1.  Open the notebook in Jupyter
2.  Run cells sequentially
3.  Provide your dataset as a Pandas DataFrame
4.  Train the model using your data

------------------------------------------------------------------------

## Notes

-   This is a basic educational implementation
-   Not optimized for production use
-   Can be extended with backpropagation, optimizers, and better
    evaluation metrics

------------------------------------------------------------------------

## Author

Week 08 Assignment Submission
