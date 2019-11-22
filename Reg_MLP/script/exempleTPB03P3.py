#-----------------------------------------------------------
# If triedpy modules are not directly accessible by python,
# set the path where triedpy is located :
if 1 :
    import sys
    TRIEDPY = "../";  # Mettre le chemin d'accès ... 
    sys.path.append(TRIEDPY);  # ... aux modules triedpy
#
import matplotlib.pyplot as plt
import TPB03_methodes
import numpy as np
#-----------------------------------------------------------
from_lr  =   0;     # Learning set starting point
to_lr    = 300;     # Learning set ending point
from_val = 300;     # Validation set starting point
to_val   = 400;     # Validation set ending point
 
lr        = 0.1;    # Learning rate (Pas d'apprentissage)
n_epochs  = 100;    # Number of whole training set presentation
plot_flag = 1;      # 1 :=> Linear plot 
                    # 2 :=> Log plot
hidden_function='tgh' #'lin'; #'lin' 'tgh'
mask_dimension = 12; # Taille du masque carre
ecf  = 10;          # Error Computation Frequency (evry ecf iterations)
gdf  = 1;           # Graphic Display Frequency (evry gdf erreur computation)
#-----------------------------------------------------------
# Les Données
x = np.loadtxt("x.txt");
t = np.loadtxt("t.txt");
#
#TPB03_methodes.display_pat(x,1,10)
# Apprentissage Avec validation ----------------------------
#np.random.seed(0);
#for i in range(5):
w1mv,b1mv,w2mv,b2mv,itmv = TPB03_methodes.trainshared(x,t,hidden_function,from_lr, \
       to_lr,from_val,to_val,mask_dimension,lr,n_epochs,plot_flag,ecf,gdf);

np.save("poids1", w1mv)
np.save("poids2", w2mv)
np.save("poidsb1", b1mv)
np.save("poidsb2", b2mv)

w1= np.load('poids1.npy')
w2= np.load('poids2.npy')
b1= np.load('poidsb1.npy')
b2= np.load('poidsb2.npy')

np.shape(w2)

np.shape(b2)

plt.matshow(w1)
plt.matshow(w2)
plt.matshow(b1)
plt.matshow(b2)

 #   print()
#
# Affichage de l'activation de la carte de caracteristique
fr=1; to=12; # indices (from, to) des formes à visualiser
TPB03_methodes.n_pattern_showing(hidden_function,w1mv,b1mv,w2mv,b2mv,x,fr,to);
#plt.title("Affichage de l'activation de la carte de caracteristique de 1 a 12 mask_dimension ",fontsize=16)