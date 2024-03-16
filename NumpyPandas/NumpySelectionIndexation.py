import numpy as np

arr = np.arange(0,11) #[ 0  1  2  3  4  5  6  7  8  9 10]

arr[1:5] #[1 2 3 4]
arr[3:] #[ 3  4  5  6  7  8  9 10]
arr[:-2] #[0 1 2 3 4 5 6 7 8]
arr[int(len(arr)/2):] #[ 5  6  7  8  9 10]

#Numpy a une particularité par rapport aux listes de python, c'est le broadcasting.
#c'est à dire qu'il est possible de modifier un ensemble d'éléments d'un np
arr      #[ 0  1  2  3  4  5  6  7  8  9 10]
arr[0:3] = 3,2,1
arr      #[3 2 1   3   4   5  6   7   8   9  10]

#en copiant simplement un tableau comme ça, si on modifie le tableau de base le nouveau le sera aussi
arrtest = arr.copy()
arrtestSimple = arr
arr[:] = 50
arrtest          #[ 3  2  1  3  4  5  6  7  8  9 10]
arrtestSimple    #[50 50 50 50 50 50 50 50 50 50 50]

arr_2d = np.array([[5,10,15],[10,20,30],[3,6,9]])   #[[ 5 10 15]
                                                    # [10 20 30]
                                                    # [ 3  6  9]]
arr_2d.shape    #(3, 3) 3 lignes et 3 colonnes

arr_2d[0]       #[ 5 10 15]

arr_2d[0][1]    #10

arr_2d[0,1]     #10

arr_2d[:2,1:]   #[[10 15]
                # [20 30]]
                
arr = np.arange(1,11)   #[ 1  2  3  4  5  6  7  8  9 10]
arr > 4                 #[False False False False  True  True  True  True  True  True]

bool_arr = arr > 4
arr[bool_arr]           #[ 5  6  7  8  9 10]

arr + 5                 #[ 6  7  8  9 10 11 12 13 14 15]
arr - 2                 #[-1  0  1  2  3  4  5  6  7  8]

arr.sum()    #55
arr.mean()   #5.5
#ecart type
arr.std()    #2.8722813232690143

arr_2d = np.arange(0,25).reshape(5,5)   #[[ 0  1  2  3  4]
                                        # [ 5  6  7  8  9]
                                        # [10 11 12 13 14]
                                        # [15 16 17 18 19]
                                        # [20 21 22 23 24]]
                                        
arr_2d.sum()                     #300
arr_2d.sum(axis=0)               #[50 55 60 65 70]    Retourne la somme des colonnes
arr_2d.sum(axis=1)               #[ 10  35  60  85 110]  Retourne la somme les lignes