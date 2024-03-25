import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
csvFile = os.path.join(CUR_DIR, "cancer_classification.csv")
df = pd.read_csv(csvFile)

#observation des correlations 
df.corr()["benign_0__mal_1"][:-1].sort_values().plot(kind="bar")
plt.show()

X = df.drop("benign_0__mal_1", axis=1).values
y = df["benign_0__mal_1"].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(units=30, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=15, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam")




if os.path.exists(os.path.join(CUR_DIR, "myModel.h5")):
    from keras.models import load_model
    model = load_model(os.path.join(CUR_DIR, "myModel.h5"))
    # Chargement de l'historique d'entraînement
    history_df = pd.read_csv(os.path.join(CUR_DIR, "training_history.csv"))

    # Affichez l'historique ou générez des graphiques à partir de celui-ci
    import matplotlib.pyplot as plt
    history_df[['loss', 'val_loss']].plot()
    plt.show()
else:
    from keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)
    model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test), callbacks=[early_stop])
    model.save(os.path.join(CUR_DIR, "myModel.h5"))
    history_dict = model.history.history

    # Utilisation de pandas pour convertir le dictionnaire en DataFrame
    history_df = pd.DataFrame(history_dict)

    # Sauvegardez le DataFrame dans un fichier CSV
    history_file_path = os.path.join(CUR_DIR, "training_history.csv")
    history_df.to_csv(history_file_path, index=False)

