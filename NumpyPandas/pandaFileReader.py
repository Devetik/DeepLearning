import pandas as pd
import numpy as np
import lxml
import os

# Importing the dataset
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
csvFile = os.path.join(CUR_DIR, "bank.csv")
dataset = pd.read_csv(csvFile)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values




# df = pd.DataFrame({'A':[1,2,np.nan,4],
#                   'B':[5,np.nan,np.nan,8],
#                   'C':[10,20,30,40]})
# df.to_json("test.json", index=False)

url = "https://en.wikipedia.org/wiki/List_of_association_footballers_who_died_after_on-field_incidents"
df = pd.read_html(url)
# url = "https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list"
# df = pd.read_html(url)
df.to_csv("swsw.csv")

print(df)
