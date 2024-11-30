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

##Dataset Description
###Features:
1.Age: Age of the customer (18-70).
2.Income: Annual income of the customer (in dollars, ranging from $20,000 to $120,000).
3.Browsing Hours: Number of hours spent browsing per day (0.5 to 10 hours).
4.Gender: Gender of the customer (Male or Female).
5.Device: Device used by the customer (Mobile, Desktop, Tablet).
6.Most Searched Item: Product category most searched by the customer (Electronics, Clothing, Home Decor, etc.).
7.Purchase: Binary outcome (0 = no purchase, 1 = purchase).
The dataset is generated synthetically, simulating real-world customer data and behaviors.

