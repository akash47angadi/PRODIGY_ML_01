# House Prices Prediction using TensorFlow Decision Forests

## Overview

This project aims to predict house prices using a baseline Random Forest model implemented with TensorFlow Decision Forests. The dataset used includes features such as square footage, number of bedrooms, and number of bathrooms. Decision Forests, a family of tree-based models including Random Forests and Gradient Boosted Trees, are utilized as they typically provide strong performance on tabular data.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Usage](#usage)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction

This project demonstrates how to train a Random Forest model using TensorFlow Decision Forests on a house price dataset. The model is trained to predict house prices based on features such as square footage, number of bedrooms, and number of bathrooms.

## Installation

Before you begin, ensure you have the following packages installed:

- TensorFlow Decision Forests
- Pandas

You can install these packages using pip:


## Dataset
The dataset should be in CSV format and include the following columns:
- SquareFootage
- Bedrooms
- Bathrooms
- Price (target variable)

Ensure your dataset is saved as `dataset.csv` in the project directory.

## Usage
Follow these steps to train and evaluate the model:

1. Import the required libraries.
2. Load the dataset.
3. Convert the dataset to a TensorFlow dataset.
4. Create and train the model.
5. Evaluate the model's performance.

```python
import tensorflow_decision_forests as tfdf
import pandas as pd

# Load the dataset
dataset = pd.read_csv("project/dataset.csv")

# Convert the dataset to a TensorFlow dataset
tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="Price")

# Create and train the model
model = tfdf.keras.RandomForestModel()
model.fit(tf_dataset)

# Print the model summary
print(model.summary())


```

## Model Training
The model is trained using the following steps:

Load the dataset into a Pandas DataFrame.
Convert the DataFrame to a TensorFlow dataset using pd_dataframe_to_tf_dataset.
Initialize a Random Forest model using tfdf.keras.RandomForestModel.
Fit the model on the training data using the fit method.

## Evaluation
After training the model, evaluate its performance using the model summary. The summary provides information about the model's structure and performance metrics.
