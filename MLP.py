# pylint: disable=all
import numpy as np  
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MLP:
    def __init__(self, n_entrées, couche_cachés,n_sorties, alpha=0.01):
        self.n_entrées = n_entrées  
        self.couche_cachés = couche_cachés  
        self.n_sorties = n_sorties  
        self.alpha = alpha  

        
        # Initialisation des poids et biais
        self.poids = []  
        self.biais = []  
        
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
    
    def entrainement(self, X, Valeurs_vrais, n_iteration=10000):
        historique_erreur = []
        for j in range(n_iteration):
            self.backward(X, Valeurs_vrais)
            if j % 500 == 0:
                sortie = self.forward(X)[0]
                erreur_moy = self.cout(sortie, Valeurs_vrais)
                historique_erreur.append(erreur_moy)
                print(f"Itération {j} : Erreur = {erreur_moy}")

        # afficher la courpe d'apprentissage
        plt.figure(figsize=(10, 5))
        plt.plot(range(0, n_iteration, 500), historique_erreur)
        plt.title("Courpe d'apprentissage")
        plt.xlabel("Itérations")
        plt.ylabel("Erreur moyenne")
        plt.grid(True)
        plt.show()
    
    def cout(self, sortie_predite,sortie_vraie):
        somme = 0
        n= sortie_predite.shape[1]
        for i in range(n):
            erreur = sortie_vraie[0,i] - sortie_predite[0, i]
            somme = somme + erreur ** 2
        return somme / n
    
    def afficher_vase(self, X, y):
        '''Afficher la avse en 3D'''
        figure = plt.figure(figsize=(10,8))
        ax = figure.add_subplot(111, projection = '3d')
        # si data en 2D , ajouter la 3éme dimension"
        if X.shape[1] == 2:
            points_vase = X[y.flatten() == 1]
            points_bruit = X[y.flatten() == 0]
            # afficher 2D 
            ax.scatter(points_vase[:, 0], points_vase[: , 1], c='red' ,s=10, label='Vase')
            ax.scatter(points_bruit[:,0], points_bruit[:,1], c='blue',s=1,alpha=0.3,label='Bruit')
        #si data en 3D
        elif X.shape[1] == 3:
            points_vase = X[y.flatten() == 1]
            points_bruit = X[y.flatten() == 0]
            #afficher end 3D
            ax.scatter(points_vase[:, 0], points_vase[: , 1],points_vase[:,2], c='red' ,s=10, label='Vase')
            ax.scatter(points_bruit[:,0], points_bruit[:,1],points_bruit[:,2], c='blue',s=1,alpha=0.3,label='Bruit')
        ax.set_title("Classification du vase en 3D")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if X.shape[1] == 3:
            ax.set_zlabel("Z")

        ax.legend()
        plt.show()
        


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
mlp_xor.entrainement(Xor, sortie, n_iteration=3000)

# Test du modèle
sortie_predite = mlp_xor.forward(Xor)[0]
print("\nSortie après entraînement (XOR) :")
print(sortie_predite)

# lir le fichier text
données = np.loadtxt(r"data/data.txt") 
print(" les  données") 
print(données[:10])  

entrées = données [:, : -1]
sorties = données[:, -1].reshape(-1, 1)  # La dernière colonne est la sortie 
print(f"Points du vase (classe 1): {np.sum(sorties == 1)}")
print(f"Points de bruit (classe 0): {np.sum(sorties == 0)}")

# Diviser les données en 80% entraînement et 20% test
X_train, X_test, y_train, y_test = train_test_split(entrées,sorties, test_size=0.2, random_state=42)  # random_state = 42 pour fixer la division des données
mlp_vase = MLP(n_entrées=X_train.shape[1], couche_cachés=[16,8,4], n_sorties=1, alpha=0.001) #entrainent
mlp_vase.entrainement(X_train.T, y_train.T, n_iteration=3000) 
mlp_vase.afficher_vase(X_train,y_train)
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
def predire_points():
    print("\nPrédictions pour les points:")
    points_predictions = []
    classes_predictions = []
    try:
        n_points = int(input("Combien de points souhaitez vous prédire ? "))
    except ValueError:
        print("Entrer un nombre entier ")
        n_points = 3
    for i in range(n_points):
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
                # pour affichage
                points_predictions.append(coordonnees)
                classes_predictions.append(predicted_class)

                # Vérifier si ce point appartient à la vase ou non
                if predicted_class == 1:
                    print(f"Le point {coordonnees} appartient à la vase.")
                else:
                    print(f"Le point {coordonnees} n'appartient pas à la vase.")
        except ValueError:
            print("Veuillez entrer des valeurs numériques valides pour les coordonnées.")
    # afficher en 3D
    if points_predictions:
        print("\nAffichage de la classification avec les points prédits")
        mlp_vase.afficher_vase(X_train,y_train,points_predictions=np.array(points_predictions),classes_predictions=np.array(classes_predictions))

predire_points()