# Microbial Community Analysis and Machine Learning Prediction for Wastewater Treatment Plants

## Project Overview

This project utilizes machine learning methods (such as Random Forest, XGBoost, Bayesian Classification, etc.) to predict environmental parameters and geographical locations of wastewater treatment plants. It focuses on analyzing microbial communities in activated sludge and identifying important OTUs (Operational Taxonomic Units) that have a significant impact on wastewater treatment through feature selection techniques. The project is divided into several parts: data processing, feature selection, model training (classification and regression), and results visualization.

## Directory Structure

The project is organized into several directories, each corresponding to different aspects of the analysis. Below is a detailed explanation of the purpose of each directory and its corresponding files:

### 1. **DATA Processing**

This folder contains scripts for data cleaning, merging, and preprocessing to prepare the dataset for model training.

- **add_.py**: Fills missing values using the K-Nearest Neighbors (KNN) algorithm.
- **count_species.py**: Counts and summarizes the bacterial species associated with each OTU in the dataset.
- **delete_zero.ipynb**: A Jupyter notebook for removing zero values and preprocessing data.
- **heat_data.py**: Generates data used to plot heatmaps, which show the relationship between microbial species and environmental variables.
- **merged.py**: Merges multiple feature files based on shared OTU IDs and fills missing values with zero.
- **N_P_pre.py** & **N_P_pre_2.py**: Predicts the removal rates of BOD, COD, nitrogen, and phosphorus based on environmental factors and microbial communities.

### 2. **Feature Selection**

This directory contains scripts for performing feature selection using various techniques such as mutual information, Spearman correlation, entropy, and more.

- **feature_C.py**: Combines multiple feature selection methods (Spearman correlation, entropy, and mutual information) and normalizes their results to select important features.
- **rf_importance.py**: Evaluates feature importance using a Random Forest model.
- **sperson.py**: Calculates Spearman correlation coefficients for feature selection and ranking.
- **XGBoost_Importance.py**: Uses XGBoost to compute feature importance, specifically for predicting environmental parameters in wastewater treatment plants.
- **Mu_in.py**: Performs feature selection using mutual information.
- **Entropy.py**: Scores features based on entropy values.


### 3. **Classification Models**

This folder contains scripts for classification tasks using different machine learning algorithms. These models are used to predict discrete variables like the geographical location of wastewater treatment plants.

- **AdaBoost_classification.py**: Classification using the AdaBoost algorithm.
- **Bayes_classification.py**: Classification using a Bayesian classifier.
- **KNN_classification.py**: Classification using the K-Nearest Neighbors (KNN) algorithm.
- **LDA_classification.py**: Linear Discriminant Analysis (LDA) for classification.
- **rf_classification.py**: Classification using the Random Forest algorithm.
- **SVM.py**: Classification using Support Vector Machines (SVM).
- **XGBoost_classification.py**: Classification using XGBoost.

### 4. **Regression Models**

This folder contains scripts for regression tasks, where the goal is to predict continuous variables such as removal rates of different pollutants in wastewater treatment.

- **BayesianRidge.py**: Bayesian Ridge Regression model.
- **DecisionTree_re.py**: Decision Tree Regression model.
- **dense_re.py**: A dense neural network-based regression model.
- **ElasticNet.py**: Elastic Net regression model.
- **GradientBoosting_re.py**: Gradient Boosting regression model.
- **KNeighbors_re.py**: K-Nearest Neighbors regression.
- **LinearRegression.py**: Linear regression model.
- **rf.py**: Random Forest regression model.
- **SVR.py**: Support Vector Regression model.
- **XGBoost.py** & **XGBoost_grid.py**: Regression using XGBoost and grid search for parameter optimization.


### 5. **Utils**

This directory contains utility scripts that assist with various tasks such as plotting, calculating correlations, and more.

- **Bayes_clfication.py**: Utility script for Bayesian classification.
- **person.py**: Script for calculating Pearson and Spearman correlation coefficients.
- **plot_confound.py**: Generates confusion matrices for classification results.
- **rf_regressor.py**: A utility script for training a Random Forest regression model.
- **scatter.py**: Generates scatter plots to visualize relationships between feature importance and Spearman correlation coefficients.
- **test2.py**: Utility script for testing classification tasks and generating importance scores for different OTUs.
