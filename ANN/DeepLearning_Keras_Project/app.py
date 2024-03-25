import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from DataPreprocessing import DataPreprocessing

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
csvFile = os.path.join(CUR_DIR, "data.csv")
df = pd.read_csv(csvFile)

DP = DataPreprocessing(df)


"""Analyse exploratoire des données"""
False and DP.analyseExploratoireData(df)

df = DP.trieData(df)
#modifier term en deux catégories
#modifier grade en catégories
#modifier sub_grade en catégories
#modifier home_ownership pour en faire deux catégories
#modifier verification_status en 3 catégories
#modifier purpose en catégories
# colonnes_categorielles = ['term', 'grade', 'sub_grade', 'home_ownership',
#                           'verification_status', 'loan_status', 'purpose',
#                           'earliest_cr_line', 'initial_list_status', 'application_type']
# df_final = pd.get_dummies(df, columns=colonnes_categorielles)
# print(df_final.head(100))

# #supprimer emp_title, trop de catégories sinon
# df = df.drop("emp_title", axis=1)
# #supprimer title, trop de catégories
# df = df.drop("title", axis=1)
# #supprimer emp_length, trop de catégories
# print(sorted(df["emp_length"].dropna().unique()))
# #supprimer adresse
# df = df.drop("address", axis=1)

# #convertir issue_d en datetime month et year puis supprimer issue_d
# df['month'] = pd.to_datetime(df['issue_d']).apply(lambda date:date.month)
# df["year"] = pd.to_datetime(df['issue_d']).apply(lambda date: date.year)
# df = df.drop("issue_d", axis=1)

# #convertir mort_acc null en 0
# df["mort_acc"] = df["mort_acc"].fillna(value=0)

# #convertir pub_rec_bankruptcies null en 0
# df["pub_rec_bankruptcies"] = df["pub_rec_bankruptcies"].fillna(value=0)
# #définir loan_status comme résultat
# # for i, row in df.iterrows():
# #     print(row["pub_rec_bankruptcies"])

# #convertir revol_util null en moyenne
# df["revol_util"] = df["revol_util"].fillna(value=df["revol_util"].mean())


# #print(df.isnull().sum())
# #print(df.describe().transpose())
# #print(df.head(50))

