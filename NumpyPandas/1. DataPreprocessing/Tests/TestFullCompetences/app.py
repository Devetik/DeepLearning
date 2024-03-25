# Import necessary libraries
import pandas as pd
import numpy as np                  # Pour la gestion des arrays sous l'allias np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load the Iris dataset
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
csvFile = os.path.join(CUR_DIR, "data.csv")
# Configure Pandas to display numbers without scientific notation
pd.set_option('display.float_format', lambda x: '%.2f' % x)
dataset = pd.read_csv(csvFile, delimiter=',')

# Separate features and target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(f"X_Data\n{X}\n\n")
print(f"y_Data\n{y}\n\n")

# # Gestion des données manquantes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, :])  # Prend en compte uniquement les valeurs numérique
X[:, :] = imputer.transform(X[:, :])

# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train\n{X_train}\n\n")
print(f"X_test\n{X_test}\n\n")
print(f"y_train\n{y_train}\n\n")
print(f"y_test\n{y_test}\n\n")

# Create an instance of the StandardScaler class
sc = StandardScaler()

# Fit the StandardScaler on the features from the training set and transform it
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)

print("Scaled training set:\n", X_train) 
print("Scaled test set:\n", X_test)





