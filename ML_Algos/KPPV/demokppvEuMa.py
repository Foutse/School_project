#-----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import sys
TRIEDPY = "../../../../";          # Chemin d'accès aux modules triedpy
sys.path.append(TRIEDPY);
from   triedpy import triedtools as tls  
import EX22_methodes               # module spécifique pour l'exercice
#
TRIEDDATA = "../EX22_enonce/";     # Chemin d'accès aux données
#-----------------------------------------------------------
# Some Conditionnements 
plt.ion(); 
#-----------------------------------------------------------
# Les données
APP  = np.loadtxt(TRIEDDATA+"DataA.txt");   # Données
TAPP = np.loadtxt(TRIEDDATA+"LabelA.txt");  # Classes
I1   = np.where(TAPP==1); # Indices des élts de la classe 1
I2   = np.where(TAPP==2); # Indices des élts de la classe 2
#
# L'ensemble de TEST constitué par les points de grille
fr=0.0; pas=0.05; to=1.5+pas;
maille = np.arange(fr, to, pas)
X, Y   = np.meshgrid(maille, maille);
X1     = np.reshape(X, np.prod(np.size(X)), 1);
Y1     = np.reshape(Y, np.prod(np.size(Y)), 1);
TEST   = np.transpose([X1, Y1]);

# Affichage des données
plt.figure();
plt.plot(APP[I1,0][0],APP[I1,1][0],'bo',markersize=6);
plt.plot(APP[I2,0][0],APP[I2,1][0],'rs',markersize=5);
plt.plot(TEST[:,0],   TEST[:,1],   'k+',markersize=8);
plt.axis("tight");
plt.legend(["C1app","C2app","test"]);
plt.title("Données d''apprentissage (2 classes) et grille de test");
#TEST
#------------------------------------------------------------
# Appliquation l'algo des k-plus proches voisins
#
# Déclaration d'un vecteur de classe arbitraire pour pouvoir 
TTEST = np.ones(np.size(TEST,1));   # utiliser le code kppvo...
#
# Choix du nombre k de voisin
k = 4;
#
#------------- Distance Euclidienne
CLA = EX22_methodes.kppvo(APP,TAPP,TEST,k);

J1 = np.where(CLA==1);    # Ici, on récupère la classification
J2 = np.where(CLA==2);    # faite par kppv sur l'ensemble de test

plt.figure();
plt.plot(TEST[J1,0][0],TEST[J1,1][0],'b+',markersize=8);
plt.plot(TEST[J2,0][0],TEST[J2,1][0],'r+',markersize=8);
plt.plot(APP[I1,0][0], APP[I1,1][0], 'bo',markersize=6);
plt.plot(APP[I2,0][0], APP[I2,1][0], 'rs',markersize=5);
plt.legend(["C1test","C2test","C1app","C2app"]);
plt.axis("tight");
plt.title("k=%d voisin(s) with euclidienne distance" %k);
#
# ------------- Distance de Mahalanobis
CLA = EX22_methodes.kppvo(APP,TAPP,TEST,k,1);
#
J1 = np.where(CLA==1);    # Ici, on récupère la classification
J2 = np.where(CLA==2);    # faite par kppv sur l'ensemble de test

plt.figure();
plt.plot(TEST[J1,0][0],TEST[J1,1][0],'b+',markersize=8);
plt.plot(TEST[J2,0][0],TEST[J2,1][0],'r+',markersize=8);
plt.plot(APP[I1,0][0], APP[I1,1][0], 'bo',markersize=6);
plt.plot(APP[I2,0][0], APP[I2,1][0], 'rs',markersize=5);
plt.legend(["C1test","C2test","C1app","C2app"]);
plt.axis("tight");
plt.title("k=%d voisin(s) with Mahalanobis distance" %k);
#
#------------------------------------------------------------
plt.show();

