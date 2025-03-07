# pylint: disable=all
import numpy as np  


class Retropropagation:
    def __init__(self, p, couche_cachés,q):
        self.p = p  # nombre d'entrées (sans le biais)
        self.couche_cachés = couche_cachés  # Liste du nombre de neurones dans chaque couche cachée
        self.q = q  # nombre de couches ( incluant la couche d'entrée et la couche de sortie)
        
        # Initialisation des poids et biais
        self.W = []  # Liste des matrices de poids
        self.b = []  # biais
        
        # Couche d'entrée  à première couche cachée
        self.W.append(np.random.randn(couche_cachés[0], p+1))
        self.b.append(np.random.randn(couche_cachés[0], 1))
        
        # (nombre de neurones dans la couche suivante) x (nombre de neurones dans la couche actuelle + 1) (le +1 est pour le biais).
        for i in range (1, len(couche_cachés)):
            self.W.append(np.random.randn(couche_cachés[i], couche_cachés[i-1]+1))
            self.b.append(np.random.randn(couche_cachés[i], 1))
        
        # Dernière couche cachée à la couche de sortie
        self.W.append(np.random.randn(1, couche_cachés[-1]+1))
        self.b.append(np.random.randn(1, 1))
    
    
    # Fonction d'activation sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Dérivée de la sigmoid 
    def derivé_sigmoid(self, x):
        return x * (1 - x)
    
    # Fonction de propagation avant
    
    