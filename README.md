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

```bash
pip install tensorflow_decision_forests pandas


Dataset
The dataset should be in CSV format and include the following columns:

SquareFootage
Bedrooms
Bathrooms
Price (target variable)
Ensure your dataset is saved as dataset.csv in the project directory.
