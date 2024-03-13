import numpy as np

np.zeros(3) #[0. 0. 0.]

np.zeros((3,3))     # [[0. 0. 0.]
                    #  [0. 0. 0.]
                    #  [0. 0. 0.]]
                    
np.ones((3,3))      # [[1. 1. 1.]
                    #  [1. 1. 1.]
                    #  [1. 1. 1.]]
                    
np.ones((3,3), dtype=int)   #[[1 1 1]
                            # [1 1 1]
                            # [1 1 1]]
                            
# linspace retourne des nombre uniformément espacé 
np.linspace(0,10,3) #[0.,5.,10.]

np.linspace(0,10,5) #[0.,2.5,5.,7.5,10.]

np.linspace(0,10,11) #[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]

#eye retourne une matrice avec une diagonale
np.eye(4)   #[[1. 0. 0. 0.]
            # [0. 1. 0. 0.]
            # [0. 0. 1. 0.]
            # [0. 0. 0. 1.]]

np.random.rand(3) #[0.97329753 0.8302968  0.97852468]

np.random.rand(3,4)     #[[0.01717359 0.25713662 0.62210282 0.28562373]        
                        # [0.01417167 0.56215163 0.19499842 0.02332063]        
                        # [0.41304621 0.38241246 0.55109089 0.9219993 ]]  

#randn permet d'avoir une variance de -1 à 1
np.random.randn(3,4)        #[[ 1.52755374 -0.94117714  0.72869724  0.56057665]    
                            # [-0.2628552  -0.1030757  0.75851913  0.6634577 ]    
                            # [ 0.61709501  0.99106472 -1.87093224 -0.1989834 ]] 
                            
a = np.random.randint(1,100) #94
np.random.randint(1,100,5) #[82 27 16  6 42]

#la seed permet de toujours avoir les mêmes nombres aléatoire
np.random.seed(42)
np.random.rand(4) #[0.37454012 0.95071431 0.73199394 0.59865848]

#création d'array de dimention 1
arr = np.arange(25) #[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
randarr = np.random.randint(0,50,10)

arr.shape #(25,)

#reshape permet de convertir un array en 1 dimention en matrice 5x5 mais ne le modifie pas de manière permanente
#il faut le réasigner pour qu'il conserve la matrice
arr.reshape(5,5)    #[[ 0  1  2  3  4]
                    # [ 5  6  7  8  9]
                    # [10 11 12 13 14]
                    # [15 16 17 18 19]
                    # [20 21 22 23 24]]
#arr.reshape(3,6) #donnera une erreur, car maximum de 18 éléments

#récupère la valeur et l'index du max
randarr.max()       #39
randarr.argmax()    #7

#récupère la valeur et l'index du min
randarr.min()       #2
randarr.argmin()    #9

a =randarr.dtype




print(a)