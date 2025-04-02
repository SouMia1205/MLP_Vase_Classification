# pylint: disable=all
import numpy as np  


class MLP:
    def __init__(self, n_entrées, couche_cachés,n_sorties, alpha=0.01):
        self.n_entrées = n_entrées  # nombre d'entrées (sans le biais)
        self.couche_cachés = couche_cachés  # Liste du nombre de neurones dans chaque couche cachée
        self.n_sorties = n_sorties  # nombre de neurones de sortie.
        self.alpha = alpha  # Définition du taux d'apprentissage

        
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
        self.poids.append(np.random.randn(n_sorties, couche_cachés[-1]))
        self.biais.append(np.random.randn(n_sorties, 1))
    
    
    # Fonction d'activation sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Dérivée de la sigmoid 
    def derivé_sigmoid(self, x):
        return x * (1 - x) 
    
    # Fonction de propagation vers l'avant (propager les entrées vers la sortie à travers le réseau)
    # ajouter variable pour renvoie toutes les valeurs intermédiaires pour retropropagation
    def forward(self, X):
        A = X
        activa = [X]
        Valeurs_Z = []
        for poids, biais in zip(self.poids, self.biais):
            Z = np.dot(poids, A) + biais
            A = self.sigmoid(Z)
            Valeurs_Z.append(Z)
            activa.append(A)
        return A, activa, Valeurs_Z
    
    def backward(self, X, Valeurs_vrais):
        valeur_predite, activa, Valeurs_Z = self.forward(X)
        erreur = Valeurs_vrais - valeur_predite
        gradient_sortie = erreur * self.derivé_sigmoid(valeur_predite)
        
        for i in range(len(self.poids) - 1, -1, -1):
            dP = np.dot(gradient_sortie, activa[i].T)
            self.poids[i] += self.alpha * dP
            dB = np.sum(gradient_sortie, axis=1, keepdims=True)
            self.biais[i] += self.alpha * dB
            if i > 0:
                gradient_sortie = np.dot(self.poids[i].T, gradient_sortie) * self.derivé_sigmoid(activa[i])
    
    def entrainement(self, X, Valeurs_vrais, n_iteration=5000):
        for j in range(n_iteration):
            self.backward(X, Valeurs_vrais)
            if j % 1000 == 0:
                erreur_moyenne = np.mean((self.forward(X)[0] - Valeurs_vrais) ** 2)
                print(f"Itération {j} : Erreur = {erreur_moyenne}")


# Exemple d'utilisation
mlp1 = MLP(n_entrées=2, couche_cachés=[3, 2], n_sorties=1, alpha=0.1)
x_iputs = np.random.randn(2, 1)  # Une entrée avec 2 features
output = mlp1.forward(x_iputs)
print("Sortie du MLP est (QST 1):", output)

mlp1.entrainement(x_iputs, np.array([[1]]), n_iteration=500)
output = mlp1.forward(x_iputs)[0]
print("Sortie après entraînement (QST 1):", output)

# Données XOR
Xor = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
sortie = np.array([[0, 1, 1, 0]])

# Initialisation du réseau
mlp_xor = MLP(n_entrées=2, couche_cachés=[4], n_sorties=1, alpha=0.1)

# Entraînement
mlp_xor.entrainement(Xor, sortie, n_iteration=5000)

# Test du modèle
sortie_predite = mlp_xor.forward(Xor)[0]
print("\nSortie après entraînement (XOR) :")
print(sortie_predite)

# lir le fichier text
données = np.loadtxt(r"C:\Users\pc\Desktop\MLP_Vase_Classification\data\data.txt") 
print(" les  données") 
print(données[:10])  

entrées = données [:, : -1]
sorties = données[:, -1].reshape(-1, 1)  # La dernière colonne est la sortie 
mlp_vase = MLP(n_entrées=entrées.shape[1], couche_cachés=[10, 5], n_sorties=1)

# Entrain du modèle sur les données
mlp_vase.entrainement(entrées.T, sorties.T, n_iteration=500)

# Tester le modèle sur les données d'entrain
sortie_predite = mlp_vase.forward(entrées.T)[0]

print("\nSortie après entraînement sur le dataset vase :")
print(sortie_predite[:10])  # Afficher les 10 premières prédictions

# Demander à l'utilisateur d'entrer les coordonnées du point séparées par des espaces
point_input = input("Entrez les coordonnées du point (séparées par des espaces) : ")

# Convertir l'entrée en une liste de nombres
point_values = np.array([float(x) for x in point_input.split()]).reshape(-1, 1)

# Vérifier si le nombre de features correspond à celui du modèle
if point_values.shape[0] != mlp_vase.n_entrées:
    print(f"Erreur : Ce modèle attend {mlp_vase.n_entrées} features, mais {point_values.shape[0]} ont été fournis.")
else:
    # Prédiction
    prediction = mlp_vase.forward(point_values)[0]

    # Seuil de classification (0.5 par défaut)
    classe = 1 if prediction >= 0.5 else 0  

    # Affichage du résultat
    print(f"Sortie brute du MLP : {prediction}")
    print(f"Classe prédite : {'Vase' if classe == 1 else 'Non Vase'}")