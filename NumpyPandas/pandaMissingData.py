import numpy as np
import pandas as pd

df = pd.DataFrame({'A':[1,2,np.nan,4],
                  'B':[5,np.nan,np.nan,8],
                  'C':[10,20,30,40]})

                    #     A    B   C
                    #0  1.0  5.0  10
                    #1  2.0  NaN  20
                    #2  NaN  NaN  30
                    #3  4.0  8.0  40

df.dropna(axis=1, thresh=3)         #     A   C
                                    #0  1.0  10
                                    #1  2.0  20
                                    #2  NaN  30
                                    #3  4.0  40

df.fillna(value="VAL ABS")          #         A        B   C
                                    #0      1.0      5.0  10
                                    #1      2.0  VAL ABS  20
                                    #2  VAL ABS  VAL ABS  30
                                    #3      4.0      8.0  40
                                    
#compléter les valeurs manquante avec la moyenne de la colonne
df["B"] = df["B"].fillna(value=df["B"].mean())  #     A    B   C
                                                #0  1.0  5.0  10
                                                #1  2.0  6.5  20
                                                #2  NaN  6.5  30
                                                #3  4.0  8.0  40

#Permet de remplacer toutes les valeurs manquante par la moyenne                                           
df = df.fillna(df.mean())                       #          A    B   C
                                                #0  1.000000  5.0  10
                                                #1  2.000000  6.5  20
                                                #2  2.333333  6.5  30
                                                #3  4.000000  8.0  40
print(df)