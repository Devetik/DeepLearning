import pandas as pd
import numpy as np
import seaborn as sns
import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
csvFile = os.path.join(CUR_DIR, "fake_reg.csv")
df = pd.read_csv(csvFile)

print(df.head())
