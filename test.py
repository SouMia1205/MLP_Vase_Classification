import random
import math
import numpy as np

class MLP:
    def __init__(self, n_entrées, couche_cachés, n_sorties):
        self.n_entrées = n_entrées  # Nombre d'entrées (sans le biais)
        self.couche_cachés = couche_cachés  # Liste du nombre de neurones dans chaque couche cachée
        self.n_sorties = n_sorties  # Nombre de neurones de sortie
        
        self.poids = []  # Liste des matrices de poids
        self.biais = []  # Liste des vecteurs de biais
        
        # Initialisation des poids et biais avec des valeurs aléatoires
        def random_matrix(rows, cols):
            return [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]
        
        self.poids.append(random_matrix(couche_cachés[0], n_entrées))
        self.biais.append(random_matrix(couche_cachés[0], 1))
        
        for i in range(1, len(couche_cachés)):
            self.poids.append(random_matrix(couche_cachés[i], couche_cachés[i-1]))
            self.biais.append(random_matrix(couche_cachés[i], 1))
        
        self.poids.append(random_matrix(n_sorties, couche_cachés[-1]))
        self.biais.append(random_matrix(n_sorties, 1))
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def derivee_sigmoid(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        A = X  # Initialisation de la sortie de la couche d'entrée
        activa = [X]
        Valeurs_Z = []
        
        for poids, biais in zip(self.poids, self.biais):
            Z = []
            for i in range(len(poids)):
                somme = biais[i][0]
                for j in range(len(poids[i])):
                    somme += poids[i][j] * A[j][0]
                Z.append([somme])
            A = [[self.sigmoid(z[0])] for z in Z]
            Valeurs_Z.append(Z)
            activa.append(A)
        
        return A, activa, Valeurs_Z
    
    def backward(self, X, Valeurs_vrais, taux_apprentissage=0.1):
        valeur_predite, activa, Valeurs_Z = self.forward(X)
        erreur = [[Valeurs_vrais[i][0] - valeur_predite[i][0]] for i in range(len(Valeurs_vrais))]
        gradient_sortie = [[erreur[i][0] * self.derivee_sigmoid(valeur_predite[i][0])] for i in range(len(erreur))]
        
        for i in range(len(self.poids) - 1, -1, -1):
            dP = [[0 for _ in range(len(self.poids[i][0]))] for _ in range(len(self.poids[i]))]
            for j in range(len(self.poids[i])):
                for k in range(len(self.poids[i][j])):
                    dP[j][k] = gradient_sortie[j][0] * activa[i][k][0]
                    self.poids[i][j][k] -= taux_apprentissage * dP[j][k]
            
            dB = [[gradient_sortie[j][0]] for j in range(len(gradient_sortie))]
            for j in range(len(self.biais[i])):
                self.biais[i][j][0] -= taux_apprentissage * dB[j][0]
            
            if i > 0:
                nouveau_gradient = []
                for j in range(len(self.poids[i][0])):  
                    somme = 0
     