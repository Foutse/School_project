#-----------------------------------------------------------
# If triedpy modules are not directly accessible by python,
# set the path where triedpy is located :
if 1 :
    import sys
    TRIEDPY = "../";  # Mettre le chemin d'accès ... 
    sys.path.append(TRIEDPY);  # ... aux modules triedpy
#
import numpy as np
import matplotlib.pyplot as plt
from   triedpy import triedtools as tls
from   triedpy    import trieddeep  as tdp
import TPB03_methodes
#-----------------------------------------------------------
# Les Données
# Choisir le fichier des données d'entrée à utiliser :
x = np.loadtxt("x.txt");
#x = np.loadtxt("hx.txt");
#x = np.loadtxt("hx_hy.txt");
#x = np.loadtxt("pb_ph.txt");
#x = np.loadtxt("pg_pd.txt");
#x = np.loadtxt("hx_hy_pb_ph.txt");
#x = np.loadtxt("hx_hy_pg_pd.txt");
# Fichier des sorties
t = np.loadtxt("t.txt");
#
#-----------------------------------------------------------
struct1     = 1;    # 1 :=> Structure lineaire sans couche cachee
                    # 2 :=> Sigmoide sans couche cachee
                    # 3 :=> Sigmoide avec une couche cachee
hidden_layer=10;    # Nombre de neuronnes caches

from_lr  =   0;     # Learning set starting point
to_lr    = 300;     # Leasrning set ending point
from_val = 300;     # Validation set starting point
to_val   = 400;     # Validation set ending point

lr        = 0.3;    # Learning rate (Pas d'apprentissage)
n_epochs  = 100000; # Number of whole training set presentation
plot_flag = 1;      # 1 :=> Linear plot 
                    # 2 :=> Log plot                 
ecf  = 10;          # Error Computation Frequency (evry ecf iterations)
gdf  = 200;         # Graphic Display Frequency (evry gdf erreur computation)

rep1 = 0;           # 0 : Initialisation aléatoire des poids
                    # 1 : Poids obtenus précédemment (en fin d'app) %=>l'archi doit etre la meme
                    # 2 : Poids initiaux précédent %=>l'archi doit etre la meme
#-----------------------------------------------------------
np.random.seed(0);
WWmv, itmv, Ytrain, Xtrain, FF = TPB03_methodes.trainingter(x,t,hidden_layer,struct1,from_lr, \
       to_lr,from_val,to_val,lr,n_epochs,plot_flag,rep1,ecf,gdf);
#-----------------------------------------------------------
TPB03_methodes.display_pat(x,1,10)
#for i in [1,2,3]:
Vrms1 =[]
Verrq1 = []
#for i in range(10):
#    for lr in [0.1,0.3,0.6,0.9,0.12, 0.15,0.18,0.21,1]:
#        for n_epochs in [100000]:
#            struct1 = 1
#            WWmv, itmv, Ytrain, Xtrain, FF1 = TPB03_methodes.trainingter(x,t,hidden_layer,struct1,from_lr, \
#            to_lr,from_val,to_val,lr,n_epochs,plot_flag,rep1,ecf,gdf);
#            Y1 = tdp.pmcout(Xtrain,WWmv,FF1)
#            errq1, rms1, = tls.errors(Ytrain, Y1,["errq","rms"])
#            Vrms1.append(rms1)
#            Verrq1.append(errq1)
#Vrms1.index(min(Vrms1))

for i in range(15):
    lr=0.01
    n_epochs = 100000
    struct1 = 1
    WWmv, itmv, Ytrain, Xtrain, FF1 = TPB03_methodes.trainingter(x,t,hidden_layer,struct1,from_lr, \
       to_lr,from_val,to_val,lr,n_epochs,plot_flag,rep1,ecf,gdf);
    Y1 = tdp.pmcout(Xtrain,WWmv,FF1)
    errq1, rms1, = tls.errors(Ytrain, Y1,["errq","rms"])
    Vrms1.append(rms1)
    Verrq1.append(errq1)
    

Vrms2 =[]
Verrq2 = []
for i in range(15):
    lr=0.04
    n_epochs = 100000
    struct1 = 2
    WWmv, itmv, Ytrain, Xtrain, FF2 = TPB03_methodes.trainingter(x,t,hidden_layer,struct1,from_lr, \
       to_lr,from_val,to_val,lr,n_epochs,plot_flag,rep1,ecf,gdf);
    Y2 = tdp.pmcout(Xtrain,WWmv,FF2)
    errq2, rms2, = tls.errors(Ytrain, Y2,["errq","rms"])
    Vrms2.append(rms2)
    Verrq2.append(errq2)

Nneurone=[1,3,5,7,9,11,13,15,17,19]
Vrms3 =[]
Verrq3 = []
struct1 = 3
lr=0.04
n_epochs = 100000
#hidden_layer =10
for hidden_layer in Nneurone:
    WWmv, itmv, Ytrain, Xtrain, FF3 = TPB03_methodes.trainingter(x,t,hidden_layer,struct1,from_lr, \
       to_lr,from_val,to_val,lr,n_epochs,plot_flag,rep1,ecf,gdf);
    Y3 = tdp.pmcout(Xtrain,WWmv,FF3)
    errq3, rms3, = tls.errors(Ytrain, Y3,["errq","rms"])
    Vrms3.append(rms3)
    Verrq3.append(errq3)
    
plt.figure();
plt.plot(Nneurone,Vrms3,'r*-',markersize=8);
plt.title("Codage pg_pd");
plt.xlabel("Numbre de neurone");



plt.plot(Nneurone,Vrms1,'g*-',markersize=8);
plt.plot(Nneurone,Vrms2,'b*-',markersize=8);
#plt.plot(Xtrain,Y1, linewidth=1.5);   
#plt.imshow(Y1)#,interpolation='none',cmap='Blues')

#TPB03_methodes.display_pat(Y,1,10)



