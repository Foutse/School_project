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
# function de pourcentage des elts bien classe
def diagonalSome(mat):
    r,t = np.shape(mat)
    somme = 0
    for i in range(1,r):
        for j in range(1,t):
            if i==j:
                somme +=mat[i,j]
    prdt = round((somme/99)*100, 2)
    return somme, prdt




# Some Conditionnements 
plt.ion(); 
#-----------------------------------------------------------
#Lecture des donnees (Data1)
APP   = np.loadtxt(TRIEDDATA+"Data1.mat");
TAPP  = np.loadtxt(TRIEDDATA+"Label1.mat");
TEST  = np.loadtxt(TRIEDDATA+"Data1test.mat");
TTEST = np.loadtxt(TRIEDDATA+"Label1test.mat");
Ntst  = np.size(TEST,0);
#

# Lecture des donnees (Data2)
#APP   = np.loadtxt(TRIEDDATA+"Data2.mat");
#TAPP  = np.loadtxt(TRIEDDATA+"Label2.mat");
#TEST  = np.loadtxt(TRIEDDATA+"Data2test.mat");
#TTEST = np.loadtxt(TRIEDDATA+"Label2test.mat");
#Ntst  = np.size(TEST,0);
# Affichage des données (par ensemble de classe)
# Indices des labels par classe de l'ens d'App
I1 = np.where(TAPP==1)[0]; # Indices des élts de la classe 1
I2 = np.where(TAPP==2)[0]; # Indices des élts de la classe 2
I3 = np.where(TAPP==3)[0]; # Indices des élts de la classe 3
# Indices des labels par classe de l'ens de Test
J1 = np.where(TTEST==1)[0]; # Indices des élts de la classe 1
J2 = np.where(TTEST==2)[0]; # Indices des élts de la classe 2
J3 = np.where(TTEST==3)[0]; # Indices des élts de la classe 3
#
plt.figure();
plt.plot(APP[I1,0],APP[I1,1],'co',markersize=6);
plt.plot(APP[I2,0],APP[I2,1],'ms',markersize=5);
plt.plot(APP[I3,0],APP[I3,1],'gd',markersize=7);
plt.plot(TEST[J1,0],TEST[J1,1],'ko',markersize=6);
plt.plot(TEST[J2,0],TEST[J2,1],'ks',markersize=5);
plt.plot(TEST[J3,0],TEST[J3,1],'kd',markersize=7);
plt.legend(['C1','C2','C3','C1','C2','C3']);
plt.title("Couleurs pleines : données d'apprentissage \nEn noir :données de test à classer");
#
#----------------------------------------------------------
for k in [1, 2, 3, 4, 5, 10, 15, 20 ]:#; # Choix du nombre de plus proches voisins à considérer
#----------------------------------------------------------
# Application de l'algo des k-plus proches voisins avec 
# la distance euclidienne
   CLA = EX22_methodes.kppvo(APP,TAPP,TEST,k);
#
# Affichage des résultats
# Indices des labels par classe Attribué par kppvo à l'ens. de Test
   K1 = np.where(CLA==1)[0]; # Indices des élts de la classe 1
   K2 = np.where(CLA==2)[0]; # Indices des élts de la classe 2
   K3 = np.where(CLA==3)[0]; # Indices des élts de la classe 3
   print("\nMatrice de confusion obtenue par kppvo avec la distance euclidienne :");
   MCTEST  = tls.matconf(TTEST,CLA);
   s,prct = diagonalSome(MCTEST)
   plt.figure();
   plt.plot(TEST[J1,0],TEST[J1,1],'ko',markersize=8);
   plt.plot(TEST[J2,0],TEST[J2,1],'ks',markersize=7);
   plt.plot(TEST[J3,0],TEST[J3,1],'kd',markersize=9);
   plt.plot(TEST[K1,0],TEST[K1,1],'co',markersize=6);
   plt.plot(TEST[K2,0],TEST[K2,1],'ms',markersize=5);
   plt.plot(TEST[K3,0],TEST[K3,1],'gd',markersize=7);
   plt.legend(['C1','C2','C3','C1','C2','C3']);
   plt.title("Classement de l'ens. de Test par k=%d plus proches voisins\navec la distance euclidienne" %k);
   plt.xlabel("pourcentage de données bien classées = %f" %prct)

#
  # print("\nMatrice de confusion obtenue par kppvo avec la distance euclidienne :");
  # MCTEST  = tls.matconf(TTEST,CLA);

#np.trace(MCTEST)
  # s,prct = diagonalSome(MCTEST)
#
#----------------------------------------------------------
# Applique l'algo des k-plus proche voisins avec 
# la distance de Mahalanobis
   CLA = EX22_methodes.kppvo(APP,TAPP,TEST,k,1);
#
# Affichage des résultats
# Indices des labels par classe Attribué par kppvo à l'ens. de Test
   K1 = np.where(CLA==1)[0]; # Indices des élts de la classe 1
   K2 = np.where(CLA==2)[0]; # Indices des élts de la classe 2
   K3 = np.where(CLA==3)[0]; # Indices des élts de la classe 3
   print("\nMatrice de confusion obtenue par kppvo avec la distance de Mahalanobis :");
   MCTEST  = tls.matconf(TTEST,CLA);
   s,prct = diagonalSome(MCTEST)

   plt.figure();
   plt.plot(TEST[J1,0],TEST[J1,1],'ko',markersize=8);
   plt.plot(TEST[J2,0],TEST[J2,1],'ks',markersize=7);
   plt.plot(TEST[J3,0],TEST[J3,1],'kd',markersize=9);
   plt.plot(TEST[K1,0],TEST[K1,1],'co',markersize=6);
   plt.plot(TEST[K2,0],TEST[K2,1],'ms',markersize=5);
   plt.plot(TEST[K3,0],TEST[K3,1],'gd',markersize=7);
   plt.legend(['C1','C2','C3','C1','C2','C3']);
   plt.title("Classement de l'ens. de Test par k=%d plus proches voisins\navec la distance de Mahalanobis" %k);
   plt.xlabel("pourcentage de données bien classées = %f" %prct)
#
 #  print("\nMatrice de confusion obtenue par kppvo avec la distance de Mahalanobis :");
 #  MCTEST  = tls.matconf(TTEST,CLA);
#
#----------------------------------------------------------
   plt.show();
