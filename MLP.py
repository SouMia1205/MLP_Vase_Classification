# pylint: disable=all
import numpy as np  


class MLP:
    def __init__(self, n_entrées, couche_cachés,n_sorties):
        self.n_entrées = n_entrées  # nombre d'entrées (sans le biais)
        self.couche_cachés = couche_cachés  # Liste du nombre de neurones dans chaque couche cachée
        self.n_sorties = n_sorties  # nombre de neurones de sortie.
        
        # Initialisation des poids et biais
        self.poids = []  # Liste des matrices de poids
        self.biais = []  # Liste des vecteurs de biais
        
        # Couche d'entrée  à première couche cachée
        self.poids.append(np.random.randn(couche_cachés[0], n_entrées))
        self.biais.append(np.random.randn(couche_cachés[0], 1))   # chaque neuron dans la première couche cachée a un biais associé.
        
        # (nombre de neurones dans la couche suivante) x (nombre de neurones dans la couche actuelle + 1) (le +1 est pour le biais).
        for i in range (1, len(couche_cachés)):
            self.poids.append(np.random.randn(couche_cachés[i], couche_cachés[i-1]))
            self.biais.append(np.random.randn(couche_cachés[i], 1))
        
        # Dernière couche cachée à la couche de sortie
        self.poids.append(np.random.randn(1, couche_cachés[-1]))
        self.biais.append(np.random.randn(n_sorties, 1))
    
    
    # Fonction d'activation sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Dérivée de la sigmoid 
    def derivé_sigmoid(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    # Fonction de propagation vers l'avant (propager les entrées vers la sortie à travers le réseau)
    def forward(self, X):
        """
        Propagation vers l'avant à travers le réseau.
        :X -- Données d'entrée (n_entrées, n_exemples)
        :return -- Sortie du réseau aprés activation.
        """
        A = X  # Initialisation de la sortie de la couche d'entrée
        for poids, biais in zip(self.poids, self.biais):
            Z = np.dot(poids, A) + biais    # Calcul linéaire  Z = WX + b
            A = self.sigmoid(Z)   # Applique la fonction d'activation sigmoid
        return A    # Sortie finale du réseau


# Example d'utilisation
mlp1 = MLP(n_entrées=2, couche_cachés=[3, 2], n_sorties=1)   # 2 entrées, 3 neurones dans la première couche cachée, 2 neurones dans la deuxième couche cachée, et 1 neurone de sortie.
x_iputs = np.random.randn(2, 1)  # Une entrée avec 2 features
output = mlp1.forward(x_iputs)
    
print("Sortie du MLP est :",output)