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
        self.best_error = float('inf')
        self.best_poids = None
        self.best_biais = None
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

                if erreur_moy < self.best_error:
                    self.best_error = erreur_moy
                    self.best_poids = [w.copy() for w in self.poids]
                    self.best_biais = [b.copy() for b in self.biais]
                    print(f"Nouveau meilleur erreur: {self.best_error}")
                    
        if self.best_poids is not None:
            self.poids = [w.copy() for w in self.best_poids]
            self.biais = [b.copy() for b in self.best_biais]
            print(f"Meilleur erreur chargée: {self.best_error}")

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
    
    def afficher_vase(self, X, y, points_predictions=None,classes_predictions=None):
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
            # afficher les points de prediction en 2D
            if points_predictions is not None and classes_predictions is not None:
                for i,point in enumerate(points_predictions):
                    if classes_predictions[i] == 1:   #appartient vase
                        ax.scatter(point[0], point[1], c='darkred',s=100,marker='*', label='Point prédit (Vase)' if i ==0 else "")
                    else: #Bruit
                        ax.scatter(point[0], point[1],c='darkblue',s=100,marker='*',label='Point prédit (Bruit)' if i == 0 else "")

        #si data en 3D
        elif X.shape[1] == 3:
            points_vase = X[y.flatten() == 1]
            points_bruit = X[y.flatten() == 0]
            #afficher end 3D
            ax.scatter(points_vase[:, 0], points_vase[: , 1],points_vase[:,2], c='red' ,s=10, label='Vase')
            ax.scatter(points_bruit[:,0], points_bruit[:,1],points_bruit[:,2], c='blue',s=1,alpha=0.3,label='Bruit')
            # afficher les points de prediction en 3D
            if points_predictions is not None and classes_predictions is not None:
                for i, point in enumerate(points_predictions):
                    if classes_predictions[i] == 1:  #vase
                        ax.scatter(point[0], point[1],point[2], c='darkred',s=100, marker='*',label='Point prédit (Vase)' if i == 0 else "")
                    else: #bruit
                        ax.scatter(point[0],point[1],point[2],c='darkblue',s=100,marker='*',label='Point prédit (Bruit)' if i == 0 else "")
        ax.set_title("Classification du vase en 3D")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if X.shape[1] == 3:
            ax.set_zlabel("Z")

        ax.legend()
        plt.show()

    def sauvegarder_poids (self, fichier):
        save_dictionnaire = {}

        for i,(p, b) in enumerate(zip(self.poids, self.biais)):
            save_dictionnaire[f'poids_{i}'] = p
            save_dictionnaire[f'biais_{i}'] = b

        np.savez(fichier, **save_dictionnaire)

    
    def charger_poids_ds_fichier(self, fichier):
        data = np.load(fichier, allow_pickle=True)
        self.poids = []
        self.biais = []

        i = 0
        while True:
            try:
                self.poids.append(data[f'poids_{i}'])
                self.biais.append(data[f'biais_{i}'])
                i += 1
            except KeyError:
                break
    
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



# Make predictions on test.txt

def predire_fichier(mlp, fichier_test, fichier_resultats=None):
    try:
        #fichier_test = r"data/test.txt"
        donnees_test = np.loadtxt(fichier_test)
        print(f"\nDonnées lues depuis {fichier_test} : ")
        for i in range(min(5, donnees_test.shape[0])):
            print(donnees_test[i])
            
        # le tableau de resultat ( features + labels (predicitons de model))
        resultats = np.zeros((donnees_test.shape[0], donnees_test.shape[1] + 1))
        for i in range(donnees_test.shape[0]):
            for j in range(donnees_test.shape[1]):
                resultats[i, j] = donnees_test[i, j]
        
        for i in range(donnees_test.shape[0]):
            point = donnees_test[i].reshape(-1,1)  # Reshape pour le format attendu par forward
            sortie = mlp.forward(point)[0] 
            # conversion en binaire
            if sortie[0, 0] > 0.5:
                classe_predite = 1
            else:
                classe_predite = 0

            resultats[i, -1] = classe_predite
        
        if fichier_resultats:
            with open(fichier_resultats, 'w') as f:
                for i in range(resultats.shape[0]):
                    valeurs = []
                    for j in range(resultats.shape[1]):
                        if j == resultats.shape[1] - 1:
                            valeurs.append(f"{int(resultats[i, j])}")
                        else:
                            valeurs.append(f"{resultats[i, j]:.8f}")
                    
                    ligne = " ".join(valeurs)
                    f.write(ligne + '\n')
            print(f"Resultats enregistrés dans le fichier {fichier_resultats}")

        print(f"\nLes 5 premieres lignes de resultats : ")
        for i in range(min(5, resultats.shape[0])):
            print(resultats[i])

                # Affichage en 3D des points de test et leurs prédictions
        print("\nAffichage en 3D des points de test avec leurs prédictions...")
        # Afficher les points de test sur la visualisation du vase
        if X_train is not None and y_train is not None:
            afficher_vase_avec_points_test(mlp, X_train, y_train, donnees_test, resultats[:, -1])
        
        return resultats

    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        return None

def afficher_vase_avec_points_test(mlp, X_train, y_train, X_test, y_pred):
    figure = plt.figure(figsize=(12, 10))
    ax = figure.add_subplot(111, projection='3d')
    
    # Afficher les données d'entraînement
    points_vase = X_train[y_train.flatten() == 1]
    points_bruit = X_train[y_train.flatten() == 0]
    
    # Afficher en 3D (pour données 3D)
    if X_train.shape[1] == 3:
        # Données d'entraînement avec alpha réduit pour voir les points de test
        ax.scatter(points_vase[:, 0], points_vase[:, 1], points_vase[:, 2], 
                   c='red', s=10, alpha=0.3, label='Vase (entraînement)')
        ax.scatter(points_bruit[:, 0], points_bruit[:, 1], points_bruit[:, 2], 
                   c='blue', s=1, alpha=0.1, label='Bruit (entraînement)')
        
        # Points de test prédits comme vase
        points_test_vase = X_test[y_pred == 1]
        if len(points_test_vase) > 0:
            ax.scatter(points_test_vase[:, 0], points_test_vase[:, 1], points_test_vase[:, 2], alpha=0.8,
                      c='red', s=40, marker='*', linewidth=3, label='Test (prédit vase)')
        
        # Points de test prédits comme bruit
        points_test_bruit = X_test[y_pred == 0]
        if len(points_test_bruit) > 0:
            ax.scatter(points_test_bruit[:, 0], points_test_bruit[:, 1], points_test_bruit[:, 2], alpha= 0.2,
                      c='lightgray', s=40, marker='o', linewidth=2, label='Test (prédit bruit)')
        
        ax.set_zlabel("Z")
    
    ax.set_title("Classification du vase avec points de test")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    plt.show()


# Exemple d'utilisation
mlp1 = MLP(n_entrées=2, couche_cachés=[3, 2], n_sorties=1, alpha=0.1)
x_iputs = np.random.randn(2, 1)  # Une entrée avec 2 features
output = mlp1.forward(x_iputs)
print("Sortie du MLP est (QST 1):", output)
mlp1.entrainement(x_iputs, np.array([[1]]), n_iteration=3000)
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

fichier_existe = False
try:
    mlp_vase.charger_poids_ds_fichier('poids_vase.npz')
    print("Chargement des poids depuis le fichier existant 'poids_vase.npz'...")
    fichier_existe = True
except:
    print("Fichier de poids non trouvé. Démarrage de l'entraînement...")
    
if not fichier_existe:
    for i in range(1):  # Utilisation de for pour une seule itération de l'entraînement
        mlp_vase.entrainement(X_train.T, y_train.T, n_iteration=3000)
        mlp_vase.sauvegarder_poids('poids_vase.npz')
        print("Poids enregistrés dans 'poids_vase.npz'")

precision = tester_modele(mlp_vase, X_test,y_test)
print(f"Exactitude ou bien Performance du modele sur les données de test : {precision:.2f}%")

# affichage du cout pour verifier l'apprentissage
sortie_entrain = mlp_vase.forward(X_train.T)[0]
erreur_entrain = mlp_vase.cout(sortie_entrain, y_train.T)
print(f"Cout final du modele apres l'entrainement est : {erreur_entrain}")

#resultats = predire_fichier(mlp_vase, r"data/test.txt", 'resultats.txt')
#print("\nAffichage des données d'entraînement pour comparaison...")


def menu_test_mlp():
    print("\n" + "="*50)
    print("MENU DE TEST DU MODÈLE MLP VASE")
    print("="*50)
    
    while True:
        print("\nChoisissez une option:")
        print("1. Tester avec des points manuels")
        print("2. Tester avec un fichier")
        print("3. Quitter")
        
        choix = input("\nVotre choix (1-3): ")
        
        if choix == "1":
            predire_points()
        elif choix == "2":
            fichier_test = input("Entrez le chemin du fichier test (défaut: data/test.txt): ").strip()
            if not fichier_test:
                fichier_test = "data/test.txt"
                
            fichier_resultats = input("Entrez le nom du fichier de résultats (défaut: resultats.txt): ").strip()
            if not fichier_resultats:
                fichier_resultats = "resultats.txt"
                
            resultats = predire_fichier(mlp_vase, fichier_test, fichier_resultats)
            print(f"Prédictions terminées et enregistrées dans {fichier_resultats}")
        elif choix == "3":
            print("Au revoir!")
            break
        else:
            print("Choix invalide. Veuillez entrer 1, 2 ou 3.")


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


menu_test_mlp()