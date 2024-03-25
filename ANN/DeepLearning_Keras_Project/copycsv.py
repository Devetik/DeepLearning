import pandas as pd
import numpy as np
import csv

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