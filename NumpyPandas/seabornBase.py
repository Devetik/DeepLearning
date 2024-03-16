import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("heart.csv")
sns.histplot(df["age"], kde=False)
print(df.head())
plt.show()
sns.countplot(x="age", data=df, hue="sex")
plt.show()

sns.boxplot(x="sex", y="age", data=df)
plt.show()

sns.scatterplot(x="chol", y="trestbps", data=df, hue="sex", size="age")
plt.show()

iris = pd.read_csv("iris.csv")
print(iris.head())
sns.pairplot(iris, hue="species")
plt.show()

#   sepal_length  sepal_width  petal_length  petal_width species
#0           5.1          3.5           1.4          0.2  setosa
#1           4.9          3.0           1.4          0.2  setosa
#2           4.7          3.2           1.3          0.2  setosa
#3           4.6          3.1           1.5          0.2  setosa
#4           5.0          3.6           1.4          0.2  setosa