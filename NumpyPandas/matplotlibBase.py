import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


X = [0,1,2,3]
y = [100,200,300,400]

plt.plot(X,y, color ="red", marker="o", markersize=15, linestyle="--")
plt.xlim(0.5,3.5)
plt.ylim(100,400)
plt.title("Mon Titre")
plt.xlabel("X Nb pièces")
plt.ylabel("Y Price")


housing = pd.DataFrame({'rooms':[1,1,2,2,2,3,3,3],
                       'price':[100,120,190,200,230,310,330,305]})

plt.scatter(housing["rooms"], housing["price"])

plt.show()