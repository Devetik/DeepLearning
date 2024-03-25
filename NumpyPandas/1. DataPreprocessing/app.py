# Data Preprocessing Template

# Importing the libraries
import numpy as np                  # Pour la gestion des arrays sous l'allias np
import matplotlib.pyplot as plt     # Pour les graphiques
import pandas as pd                 # Gère le dataset et la matrice

# Importing the dataset
dataset = pd.read_csv('DataPreprocessing/Data.csv') # Création d'un dataframe à partir des data du csv
X = dataset.iloc[:, :-1].values #iloc est une fonction qui permet de récupérer l'index des rows puis des colonnes
                                # : permet de prendre toutes les rows puis :-1 permet de prendre toutes les colonnes sauf la dernière
                                # X pour les featured variable. [:, :-1] définis l'index des colones qu'on prends
                                # [:, :-1] prends toutes les colonnes sauf la dernière
                                # .values permet de prendre toutes les valeurs
y = dataset.iloc[:, -1].values  # y pour la dependant variable. [:, -1] prends aucunes colones sauf la dernière
#print(X)
#print(y)

# Gestion des données manquantes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')# strategy='mean' dit qu'on utilise la moyenne, 
                                                               # mais on pourrait aussi utiliser la valeur median
imputer.fit(X[:, 1:3])  # Prend en compte uniquement les valeurs numérique
X[:, 1:3] = imputer.transform(X[:, 1:3])
#print(X)



# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
#print(X)
# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)  # type: ignore
#print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print(f"X_train\n{X_train}\n\n")
print(f"X_test\n{X_test}\n\n")
print(f"y_train\n{y_train}\n\n")
print(f"y_test\n{y_test}\n\n")

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(f"X_train\n{X_train}\n\n")
print(f"X_test\n{X_test}\n\n")

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)