import numpy as np
import pandas as pd

labels = ['a','b','c']
my_list = [10,20,30]
arr = np.array([10,20,30])
d = {'a':10, 'b':20, 'c':30}

pd.Series(data=my_list, index=labels)           #a    10     
                                                #b    20     
                                                #c    30 
                                                #dtype: int64
                                                
print(pd.Series(d))                             #a    10     
                                                #b    20     
                                                #c    30 
                                                #dtype: int64
                                                

sales_Q1 = pd.Series(data=[250,450,200,150], index=['USA', 'France', 'Switzerland', 'Russia'])
sales_Q2 = pd.Series(data=[350,225,360,150], index=['USA', 'Japan', 'Switzerland', 'Russia'])
print(sales_Q1 + sales_Q2)      #France           NaN       
                                #Japan            NaN       
                                #Russia         300.0       
                                #Switzerland    560.0       
                                #USA            600.0       
                                #dtype: float64

columns = ["w", "x", "Y", "Z"]
index = ["A", "B", "C", "D", "E"]
from numpy.random import randint
np.random.seed(42)
data = randint(-100,100,(5,4))
print(data)

df = pd.DataFrame(data,index,columns)   
print(df)                               #    w   x   Y   Z
                                        #A   2  79  -8 -86
                                        #B   6 -29  88 -80
                                        #C   2  21 -26 -13
                                        #D  16  -1   3  51
                                        #E  30  49 -48 -99
                        
print(df[["w", "Z"]])                   #    w   Z
                                        #A   2 -86
                                        #B   6 -80
                                        #C   2 -13
                                        #D  16  51
                                        #E  30 -99

#Ajouter une colonne
df["SUM"] = df["w"] + df["x"] + df["Y"] + df["Z"]       #    w   x   Y   Z  SUM     
                                                        #A   2  79  -8 -86  -13     
                                                        #B   6 -29  88 -80  -15     
                                                        #C   2  21 -26 -13  -16     
                                                        #D  16  -1   3  51   69     
                                                        #E  30  49 -48 -99  -68 
 
#Supprimer une colonne                                                     
df = df.drop("SUM", axis=1)                             #    w   x   Y   Z
                                                        #A   2  79  -8 -86
                                                        #B   6 -29  88 -80
                                                        #C   2  21 -26 -13
                                                        #D  16  -1   3  51
                                                        #E  30  49 -48 -99

#Récupère une ligne
print(df.loc[["A", "E"]])                               #    w   x   Y   Z
                                                        #A   2  79  -8 -86
                                                        #E  30  49 -48 -99

#Récupère la valeur à un index spécifié
print(df.loc["A", "w"])                 #2

print(df.loc[["A", "D"],"w"])           #A     2                 
                                        #D    16
                                
print(df.loc[["A", "D"],["w", "Z"]])    #    w   Z
                                        #A   2 -86
                                        #D  16  51
                                        
print(df > 0)                           #      w      x      Y      Z
                                        #A  True   True  False  False
                                        #B  True  False   True  False
                                        #C  True   True  False  False
                                        #D  True  False   True   True
                                        #E  True   True  False  False

print(df[ df > 0 ])                     #    w     x     Y     Z
                                        #A   2  79.0   NaN   NaN
                                        #B   6   NaN  88.0   NaN
                                        #C   2  21.0   NaN   NaN
                                        #D  16   NaN   3.0  51.0
                                        #E  30  49.0   NaN   NaN
                                        
print(df["x"] > 0)                      #A     True
                                        #B    False
                                        #C     True
                                        #D    False
                                        #E     True

#Permet de faire une séléection conditionnel                                        
print(df[df["x"] > 0])                  #    w   x   Y   Z
                                        #A   2  79  -8 -86
                                        #C   2  21 -26 -13
                                        #E  30  49 -48 -99

#Permet de sélectionner la colonne w où la colonne x est plus grande que 0
print(df[df["x"] > 0]["w"])             #A     2
                                        #C     2
                                        #E    30
                                        
print(df[(df["w"] > 0) & (df["Y"] > 0)])    #    w   x   Y   Z
                                            #B   6 -29  88 -80
                                            #D  16  -1   3  51
                                            
print(df[(df["w"] > 0) | (df["Y"] > 0)])    #    w   x   Y   Z
                                            #A   2  79  -8 -86
                                            #B   6 -29  88 -80
                                            #C   2  21 -26 -13
                                            #D  16  -1   3  51
                                            #E  30  49 -48 -99
                                            
new_index = ["CH", "FR", "RU", "GE", "US"]
df["States"] = new_index

df = df.set_index("States")         #States
                                    #CH       2  79  -8 -86
                                    #FR       6 -29  88 -80
                                    #RU       2  21 -26 -13
                                    #GE      16  -1   3  51
                                    #US      30  49 -48 -99
                                    
print(df.describe())                #              w          x          Y          Z
                                    #count   5.00000   5.000000   5.000000   5.000000
                                    #mean   11.20000  23.800000   1.800000 -45.400000
                                    #std    11.96662  42.109381  51.915316  63.366395
                                    #min     2.00000 -29.000000 -48.000000 -99.000000
                                    #25%     2.00000  -1.000000 -26.000000 -86.000000
                                    #50%     6.00000  21.000000  -8.000000 -80.000000
                                    #75%    16.00000  49.000000   3.000000 -13.000000
                                    #max    30.00000  79.000000  88.000000  51.000000