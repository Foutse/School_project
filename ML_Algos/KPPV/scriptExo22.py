# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 07:52:55 2018
"""
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
# On construire les donnees
#x1=(2,2), x2=(2,-2), x3=(-2,-2) qui sont de classe 1 et
#x4=(-2,2), x5=(0,0) qui sont de classe 2.

APP  = np.loadtxt(TRIEDDATA+"datapoints.txt");   # Données
#TAPP = np.loadtxt(TRIEDDATA+"labelpoints.txt");  # Classes
TAPP = np.loadtxt(TRIEDDATA+"labelsinversed.txt");  # Classes
I1   = np.where(TAPP==1); # Indices des élts de la classe 1
I2   = np.where(TAPP==2); # Indices des élts de la classe 2
#
# L'ensemble de TEST constitué par les points de grille
fr=-3; pas=0.2; to=3+pas;
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
plt.legend(["C1","C2","test"]);
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
for k in [1,2,3,4,5] :
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
    plt.title("k=%d voisin(s) with euclidienne distance avec label inverse" %k);


#cas ou le max et le min sont egaux
#------------- Distance Euclidienne aleatoire----
for k in [1,2,3,4,5] :
   CLA = EX22_methodes.kppvo_a(APP,TAPP,TEST,k);

   J1 = np.where(CLA==1);    # Ici, on récupère la classification
   J2 = np.where(CLA==2);    # faite par kppv sur l'ensemble de test

   plt.figure();
   plt.plot(TEST[J1,0][0],TEST[J1,1][0],'b+',markersize=8);
   plt.plot(TEST[J2,0][0],TEST[J2,1][0],'r+',markersize=8);
   plt.plot(APP[I1,0][0], APP[I1,1][0], 'bo',markersize=6);
   plt.plot(APP[I2,0][0], APP[I2,1][0], 'rs',markersize=5);
   plt.legend(["C1test","C2test","C1app","C2app"]);
   plt.axis("tight");
   plt.title("k=%d voisin(s) with euclidienne distance en prenant compte du cas aleatoire label inverse" %k);