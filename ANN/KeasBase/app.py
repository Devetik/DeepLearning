import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
csvFile = os.path.join(CUR_DIR, "fake_reg.csv")
df = pd.read_csv(csvFile)

#séparation de nos données d'entrainement et de test
from sklearn.model_selection import train_test_split
X = df[["feature1", "feature2"]].values
y = df["price"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#normalization des datas
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense
#pour la création du model, nous avons deux syntaxes similaire
#model = Sequential([Dense(4, activation="relu"), Dense(2, activation="relu"), Dense(1)])
model = Sequential()
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(1))

model.compile(optimizer="rmsprop", loss="mse")

model.fit(x=X_train, y=y_train, epochs=250)

loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
plt.show()

print(f"X_train, y_train: {model.evaluate(X_train, y_train, verbose=0)}")   #X_train, y_train: 25.25990867614746
print(f"X_test, y_test: {model.evaluate(X_test, y_test, verbose=0)}")       #X_test, y_test: 25.57649040222168

test_predictions = model.predict(X_test)
test_predictions = pd.Series(test_predictions.reshape(300,))
pred_df = pd.DataFrame(y_test, columns=["Test y"])
pred_df = pd.concat([pred_df, test_predictions], axis=1)
pred_df.columns = ["Test Y", "Model Prediction"]
sns.scatterplot(x="Test Y", y="Model Prediction", data=pred_df)
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error

print(f"mean_absolute_error = {mean_absolute_error(pred_df['Test Y'], pred_df['Model Prediction'])}")
print(f"mean_squared_error**0.5 = {mean_squared_error(pred_df['Test Y'], pred_df['Model Prediction'])**0.5}")


#predir le prix avec de nouvelles données
new_gem = [[998,1000]]
new_gem = scaler.transform(new_gem)
print(model.predict(new_gem))

#sauvegarder notre model
from keras.models import load_model
model.save(os.path.join(CUR_DIR, "myModel.h5"))

#charger le model
chargerModel = load_model(os.path.join(CUR_DIR, "myModel.h5"))
print(f"Estimated price on the new gem: {chargerModel.predict(new_gem)}")