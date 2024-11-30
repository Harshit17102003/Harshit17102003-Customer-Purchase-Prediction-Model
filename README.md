# Customer Purchase Prediction Model

## Overview
This project implements a **Logistic Regression** model to predict whether a customer will purchase a product based on their browsing history, demographics, and most searched items. The model is built on a synthetic dataset containing variables such as **Age**, **Income**, **Browsing Hours**, **Gender**, **Device**, and **Most Searched Item**.

### Key Features:
- **Customer Demographics**: Age, Income, Gender, Device.
- **Browsing Behavior**: Browsing hours and Most Searched Item.
- **Prediction**: Binary classification to predict the likelihood of a customer making a purchase.

## Table of Contents
1. [Installation](#installation)
2. [Dataset Description](#dataset-description)
3. [Model Overview](#model-overview)
4. [Running the Model](#running-the-model)
5. [Key Performance Indicators (KPIs)](#key-performance-indicators-kpis)
6. [Visualizations](#visualizations)
7. [Future Work](#future-work)

---

## Installation

### Prerequisites
To run the code, you need to have the following Python libraries installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `seaborn`
- `matplotlib`

You can install them using `pip`:

pip install pandas numpy scikit-learn seaborn matplotlib

# Dataset Description

## Features:
1. **Age**: Age of the customer (18-70).
2. **Income**: Annual income of the customer (in dollars, ranging from $20,000 to $120,000).
3. **Browsing Hours**: Number of hours spent browsing per day (0.5 to 10 hours).
4. **Gender**: Gender of the customer (Male or Female).
5. **Device**: Device used by the customer (Mobile, Desktop, Tablet).
6. **Most Searched Item**: Product category most searched by the customer (Electronics, Clothing, Home Decor, etc.).
7. **Purchase**: Binary outcome (0 = no purchase, 1 = purchase).

The dataset is generated synthetically, simulating real-world customer data and behaviors.

# Model Overview

The project uses **Logistic Regression**, a binary classification algorithm, to predict customer purchases.

### Steps:
1. **Data Preprocessing**:  
   The dataset is cleaned, and categorical variables (Gender, Device, Most Searched Item) are encoded into numerical values.
2. **Model Training**:  
   The dataset is split into training and testing sets. The logistic regression model is trained on the training data.
3. **Evaluation**:  
   The model's accuracy and classification metrics (precision, recall, F1-score) are computed.
4. **Visualization**:  
   Key insights are visualized to understand customer behavior and model performance.

---

# Running the Model

### 1. Prepare the Dataset:  
The `customer_data.csv` file is generated using synthetic data creation code.

**Example Command**:
 python generate_data.py

### 2. Train the Model:
After generating the dataset, run the `train_model.py` script to train the logistic regression model.

**Example Command**:
python train_model.py

### 3. Evaluate the Model:
After training, the model’s accuracy and classification report will be displayed. It will also output key performance metrics such as `Precision`, `Recall`, `F1-Score`, and `Confusion Matrix`.

## Key Performance Indicators (KPIs)

The following KPIs are important to evaluate the model's effectiveness and answer business questions:

### 1. Purchase Rate by Age Group
How does the likelihood of purchasing vary across different age groups?

### 2. Impact of Browsing Hours on Purchases
Does the number of hours spent browsing correlate with the likelihood of making a purchase?

### 3. Effect of Device Type on Purchases
Are customers using mobile devices less likely to make a purchase compared to desktop users?

### 4. Impact of Most Searched Items on Purchases
How does the most searched product category influence purchasing behavior?

These KPIs help evaluate the model's performance and provide actionable insights for business strategies.


## Visualizations

1. **Correlation Heatmap**:  
   Visualizes the correlation between different features and the target variable (`Purchase`), helping to identify the most influential factors.

2. **Purchase Rate by Age Group**:  
   A bar chart showing the average purchase rate across different age groups. This helps understand which demographic segments are more likely to purchase.

3. **Browsing Hours vs. Purchase**:  
   A scatter plot to analyze how browsing time correlates with the likelihood of making a purchase.

4. **Purchase Rate by Most Searched Item**:  
   A bar chart showing the purchase rate for customers based on their most searched product category.

## Future Work

1. **Improve Data Balance**:  
   The current dataset has a skewed distribution (80% non-purchase, 20% purchase). Techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) can be applied to balance the dataset.

2. **Hyperparameter Tuning**:  
   The logistic regression model can be optimized using **cross-validation** and **grid search** for better performance.

3. **Incorporate More Features**:  
   Additional features like **location**, **time of day**, or **browsing history** (number of items viewed) could improve the model.

4. **Deploy the Model**:  
   This model could be deployed as an **API** for real-time prediction of customer purchases.





 





