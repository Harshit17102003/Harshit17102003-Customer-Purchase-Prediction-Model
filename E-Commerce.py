import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Step 1: Load the dataset
df = pd.read_csv("customer_data.csv")
print(df.describe,"\n")
print(df.head(5),"\n")
print(df.info())

# Step 2: Preprocess the Data
# Encode categorical variables
df = pd.get_dummies(df, columns=["Gender", "Device", "Most_Searched_Item"], drop_first=True)

# Separate features and target variable
X = df.drop("Purchase", axis=1)
y = df["Purchase"]

# Step 3: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
print("Accuracy : ", accuracy_score(y_test, y_pred))
print("F1_Score : ", f1_score(y_test, y_pred))
print("\nClassification Report : \n", classification_report(y_test, y_pred))
print("\nConfusion Matrix : \n", confusion_matrix(y_test, y_pred))

# Step 7: Visualizations
# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Feature Correlation Heatmap", fontweight = 'bold')
plt.show()

# Purchase Rate by Age Group
df['Age_Group'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 70], labels=["18-25", "26-35", "36-45", "46-55", "56-70"])
age_purchase = df.groupby("Age_Group")["Purchase"].mean()
age_purchase.plot(kind="bar", color="skyblue", figsize=(8, 5))
plt.title("Purchase Rate by Age Group", fontweight = 'bold')
plt.xlabel("Age Group")
plt.ylabel("Purchase Rate")
plt.xticks(rotation=0)
plt.show()

# Purchase Probability vs. Browsing Hours
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Browsing_Hours", y="Purchase", alpha=0.5)
plt.title("Browsing Hours vs. Purchase", fontweight = 'bold')
plt.xlabel("Browsing Hours")
plt.ylabel("Purchase (0 or 1)")
plt.show()

# Impact of Most Searched Items on Purchases
searched_items_purchase = df.groupby("Most_Searched_Item_Electronics")["Purchase"].mean()
searched_items_purchase.plot(kind="bar", color="coral", figsize=(10, 6))
plt.title("Purchase Rate by Most Searched Items", fontweight = 'bold')
plt.xlabel("Most Searched Items")
plt.ylabel("Purchase Rate")
plt.xticks(rotation=45)
plt.show()