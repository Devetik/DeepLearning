import numpy as np
import pandas as pd

df_one = pd.DataFrame({'k1':['A','A','B','B','C','C'],
                      'col1':[100,200,300,300,400,500],
                      'col2':['NY','CA','WA','WA','AK','NV']})

print(df_one["col2"].unique())          #['NY' 'CA' 'WA' 'AK' 'NV']

print(df_one["col2"].nunique())         #5

print(df_one["col2"].value_counts())    #col2
                                        #WA    2
                                        #NY    1
                                        #CA    1
                                        #AK    1
                                        #NV    1

df_one["NEW"] = df_one["col1"] * 10     #  k1  col1 col2   NEW
                                        #0  A   100   NY  1000
                                        #1  A   200   CA  2000
                                        #2  B   300   WA  3000
                                        #3  B   300   WA  3000
                                        #4  C   400   AK  4000
                                        #5  C   500   NV  5000

def grab_first_letter(state):
    return state[0]

print(grab_first_letter("NY")) #N

#ne pas mettre les paranthèses afin que la fonction s'applique partout
df_one["first letter"] = df_one["col2"].apply(grab_first_letter)
                                        #  k1  col1 col2   NEW first letter
                                        #0  A   100   NY  1000            N
                                        #1  A   200   CA  2000            C
                                        #2  B   300   WA  3000            W
                                        #3  B   300   WA  3000            W
                                        #4  C   400   AK  4000            A
                                        #5  C   500   NV  5000            N

def is_in_washington(state):
    if state[0] == "W":
        return "Washington"
    else:
        return "Error"

df_one["first letter"] = df_one["col2"].apply(is_in_washington)
                                        #  k1  col1 col2   NEW first letter
                                        #0  A   100   NY  1000        Error
                                        #1  A   200   CA  2000        Error
                                        #2  B   300   WA  3000   Washington
                                        #3  B   300   WA  3000   Washington
                                        #4  C   400   AK  4000        Error
                                        #5  C   500   NV  5000        Error
                                        
#Il est possible de mapper les valeurs, c'est à dire faire des liens entre elles                                      
my_map = {"A":11, "B":22, "C":33}
print(df_one["k1"].map(my_map))         # 0    11
                                        # 1    11
                                        # 2    22
                                        # 3    22
                                        # 4    33
                                        # 5    33

#Ajout d'une colonne avec mapping
df_one["num"] = df_one["k1"].map(my_map)    #   k1  col1 col2   NEW first letter  num
                                            # 0  A   100   NY  1000        Error   11
                                            # 1  A   200   CA  2000        Error   11
                                            # 2  B   300   WA  3000   Washington   22
                                            # 3  B   300   WA  3000   Washington   22
                                            # 4  C   400   AK  4000        Error   33
                                            # 5  C   500   NV  5000        Error   33
  
#retourne la valeur max                                          
df_one["col1"].max()                        #500
#retourne l'index de la valeur max
df_one["col1"].idxmax()                     #5

#Permet de renommer les colonnes
df_one.columns = ["C1", "C2", "C3", "C4", "C5", "C6"]
                                            #  C1   C2  C3    C4          C5  C6
                                            #0  A  100  NY  1000       Error  11
                                            #1  A  200  CA  2000       Error  11
                                            #2  B  300  WA  3000  Washington  22
                                            #3  B  300  WA  3000  Washington  22
                                            #4  C  400  AK  4000       Error  33
                                            #5  C  500  NV  5000       Error  33

#Permet de supprimer des colonnes
df_one = df_one.drop(["C5","C6"], axis=1)   #  C1   C2  C3    C4
                                            #0  A  100  NY  1000
                                            #1  A  200  CA  2000
                                            #2  B  300  WA  3000
                                            #3  B  300  WA  3000
                                            #4  C  400  AK  4000
                                            #5  C  500  NV  5000

df_one = df_one.sort_values("C3")           #  C1   C2  C3    C4
                                            #4  C  400  AK  4000
                                            #1  A  200  CA  2000
                                            #5  C  500  NV  5000
                                            #0  A  100  NY  1000
                                            #2  B  300  WA  3000
                                            #3  B  300  WA  3000
                                            
                                            
features = pd.DataFrame({'A':[100,200,300,400,500],'B':[12,13,14,15,16]})
                                            #     A   B
                                            #0  100  12
                                            #1  200  13
                                            #2  300  14
                                            #3  400  15
                                            #4  500  16

predictions = pd.DataFrame({'pred':[0,1,1,0,1]})
                                            #   pred
                                            #0     0
                                            #1     1
                                            #2     1
                                            #3     0
                                            #4     1
                                            
print(pd.concat([features, predictions], axis=1))
                                            #     A   B  pred
                                            #0  100  12     0
                                            #1  200  13     1
                                            #2  300  14     1
                                            #3  400  15     0
                                            #4  500  16     1

test = pd.get_dummies(df_one["C1"])
                                            #4  False  False   True
                                            #1   True  False  False
                                            #5  False  False   True
                                            #0   True  False  False
                                            #2  False   True  False
                                            #3  False   True  False
print(test)



