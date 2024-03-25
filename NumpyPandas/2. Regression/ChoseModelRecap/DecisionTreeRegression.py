# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Importing the dataset
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
csvFile = os.path.join(CUR_DIR, "Data.csv")
dataset = pd.read_csv(csvFile)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

## Training the Decision Tree Regression model on the Training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

## Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

## Evaluating the Model Performance
from sklearn.metrics import r2_score
R2_Score = r2_score(y_test, y_pred)
print(f"Decision Tree\nLe score R2 de ce model est de: {R2_Score}")