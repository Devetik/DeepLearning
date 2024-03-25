import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import csv


class DataPreprocessing:
    def __init__(self, DataFrame):
        self.DataFrame = DataFrame
        
    def getFullCSV(self):
        list_url = [
            'https://raw.githubusercontent.com/moncoachdata/DATA_DEEP_LEARNING/master/lending_club_loan_two-1.csv',
            'https://raw.githubusercontent.com/moncoachdata/DATA_DEEP_LEARNING/master/lending_club_loan_two-2.csv',
            'https://raw.githubusercontent.com/moncoachdata/DATA_DEEP_LEARNING/master/lending_club_loan_two-3.csv',
            'https://raw.githubusercontent.com/moncoachdata/DATA_DEEP_LEARNING/master/lending_club_loan_two-4.csv',
            'https://raw.githubusercontent.com/moncoachdata/DATA_DEEP_LEARNING/master/lending_club_loan_two-5.csv',
            'https://raw.githubusercontent.com/moncoachdata/DATA_DEEP_LEARNING/master/lending_club_loan_two-6.csv'
            ]

        df = pd.concat([pd.read_csv(f, dtype=str) for f in list_url], ignore_index=True)
        df.to_csv("data.csv", quoting=csv.QUOTE_NONNUMERIC, index=False)
        
    def analyseExploratoireData(self, df):
        sns.countplot(x="loan_status", data=df)
        plt.show()
        plt.figure(figsize=(12,4))
        sns.distplot(df["loan_amnt"], kde=False, bins=40)
        plt.show()
        # sns.heatmap(df.corr(), annot=True, cmap="viridis")
        # plt.ylim(10,0)
        # plt.show()
    
    def trieData(self, df):
        #supprimer emp_title, trop de catégories sinon
        df = df.drop("emp_title", axis=1)
        
        emp_length_order = [ '< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
        if True:
            plt.figure(figsize=(12,4))
            sns.countplot(x="emp_length", data=df, order=emp_length_order)
            plt.show()
            sns.countplot(x="emp_length", data=df, order=emp_length_order, hue="loan_status")
            plt.show()
            #Déterminer pour chaque tranche, la quantité de personnes ainsi que le pourcentage qu'ils représentent
            # emp_co = df[df["load_status"]=="Charged Off"].groupby("emp_length").count()["loan_status"]
            # emp_fp = df[df["load_status"]=="Fully Paid"].groupby("emp_length").count()["loan_status"]
            # ratio = emp_co / emp_fp
        df = df.drop("emp_length", axis = 1)
        df = df.drop("title", axis=1)
        return df
        