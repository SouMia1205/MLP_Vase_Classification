# pylint: disable=all
import numpy as np  
from sklearn.model_selection import train_test_split


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
            Z = np.dot(poids, A) + biais    # Z = W*A + b
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
            dB = np.sum(gradient_sortie, axis=1, keepdims=True)  #La somme est effectuée sur les colonnes keepdims changer [0.6, 0.3] --> [[0.6], [0.3]] (format)
            self.biais[i] += self.alpha * dB
            if i > 0:
                gradient_sortie = np.dot(self.poids[i].T, gradient_sortie) * self.derivé_sigmoid(activa[i])
    
    def entrainement(self, X, Valeurs_vrais, n_iteration=5000):
        for j in range(n_iteration):
            self.backward(X, Valeurs_vrais)
            if j % 1000 == 0:
                sortie = self.forward(X)[0]
                erreur_moy = self.cout(sortie, Valeurs_vrais)
                print(f"Itération {j} : Erreur = {erreur_moy}")
    
    def cout(self, sortie_predite,sortie_vraie):
        somme = 0
        n= sortie_predite.shape[1]
        for i in range(n):
            erreur = sortie_vraie[0,i] - sortie_predite[0, i]
            somme = somme + erreur ** 2
        return somme / n


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

# Diviser les données en 80% entraînement et 20% test
X_train, X_test, y_train, y_test = train_test_split(entrées,sorties, test_size=0.2, random_state=42)  # random_state = 42 pour fixer la division des données
mlp_vase = MLP(n_entrées=X_train.shape[1], couche_cachés=[20, 10], n_sorties=1, alpha=0.01) #entrainent
mlp_vase.entrainement(X_train.T, y_train.T, n_iteration=5000) 

# affichage du cout pour verifier l'apprentissage
sortie_entrain = mlp_vase.forward(X_train.T)[0]
erreur_entrain = mlp_vase.cout(sortie_entrain, y_train.T)
print(f"Cout final du modele apres l'entrainement est : {erreur_entrain}")

def tester_modele(mlp,  X_test,y_test):
    sortie_predite = mlp.forward(X_test.T)[0]
    # arrondir les sorties pour des predictions binairs
    sortie_binaire = (sortie_predite > 0.5).astype(int)   
    predictions_vrais = 0
    total_predictions = y_test.T.shape[1]
    # Comaraiosn de chaque prediction
    for i in range(total_predictions):
        if sortie_binaire[0, i] == y_test.T[0,i]:
            predictions_vrais +=1
    # Calculer la precision
    precision = (predictions_vrais/total_predictions) *100
    return precision

precision = tester_modele(mlp_vase, X_test,y_test)
print(f"Exactitude ou bien Performance du modele sur les données de test : {precision:.2f}%")



# Demander à l'utilisateur de saisir les coordonnées du point (3 points par example)
for i in range(3):
    try:
        coordonnees = input(f"[{i+1}]Veuillez entrer les coordonnées du point (séparées par des espaces) : ")
        coordonnees = [float(x) for x in coordonnees.split()]
        
        # Vérifier si le nombre de coordonnées est correct
        if len(coordonnees) != entrées.shape[1]:
            print(f"Le nombre de coordonnées n'est pas correct. Ce modèle attend {entrées.shape[1]} coordonnées.")
        else:
            #  tester le point
            point_test = np.array([coordonnees])  
            sortie_point = mlp_vase.forward(point_test.T)[0]  # Prédiction pour ce point

            # Arrondir la sortie pour obtenir la prédiction binaire
            predicted_class = 1 if sortie_point > 0.5 else 0

            # Vérifier si ce point appartient à la vase ou non
            if predicted_class == 1:
                print(f"Le point {coordonnees} appartient à la vase.")
            else:
                print(f"Le point {coordonnees} n'appartient pas à la vase.")
    except ValueError:
        print("Veuillez entrer des valeurs numériques valides pour les coordonnées.")