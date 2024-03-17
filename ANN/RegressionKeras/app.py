import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
csvFile = os.path.join(CUR_DIR, "kc_house_data.csv")
df = pd.read_csv(csvFile)

#Affiche le nombre de valeur NULL par colonne
print(df.isnull().sum())

non_hight = df.sort_values("price", ascending=False).iloc[216:]
#sns.scatterplot(x="long", y="lat", data=non_hight, hue="price", edgecolor=None, alpha=0.2, palette="RdYlGn")
#plt.show()

#suppression des données non pertinentes
df = df.drop("id", axis=1)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date:date.month)
df["year"] = df["date"].apply(lambda date: date.year)
df = df.drop("date", axis=1) #suppression de la date
print(df["zipcode"].value_counts())#permet d'afficher le nombre de zipcode différent
df = df.drop("zipcode", axis=1)#supprimé car 70 dummies différent à ajouter sinon

#si on veut grouper le nombre de vente par mois
#df.groupby("month").mean()["price"].plot()

X = df.drop("price", axis=1).values
y = df["price"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_trin = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(19, activation="relu"))
model.add(Dense(19, activation="relu"))
model.add(Dense(19, activation="relu"))
model.add(Dense(19, activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")


model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), batch_size=128, epochs=400)
