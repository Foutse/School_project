from TPC04_methodes import *
import numpy as np
import matplotlib.pyplot as plt
import triedctk as ctk
from   matplotlib import cm

data=np.loadtxt('el_nino.mat')
Xapp=data[108:,1:5]
sst1=data[108:,1]
sst2=data[108:,2]
sst3=data[108:,3]
sst4=data[108:,4]

def apprentissage(nlmap, ncmap,it,Tmax,Tmin) :
    '''
    % Code pour l'apprentissage en 2 phases de la Carte
    
    En sortie :
    sMap       : La structure de la carte
    Xapp : L'ensemble d'apprentissage 
    
    '''
    from   triedpy import triedsompy as SOM
    # Positionnement du rand pour avoir ou pas le meme alea a chaque fois
    seed = 0; 
    np.random.seed(seed);
    
    # Creation d'une structure de carte initialisee (référents non initialisés)
    initmethod='random'; # 'random', 'pca'
    sMap  = SOM.SOM('sMap', Xapp, mapsize=[nlmap, ncmap], norm_method='data', \
                  initmethod=initmethod)

    # APPRENTISSAGE 
    #------------------------------------------------------------------
    tracking = 'on';  # niveau de suivi de l'apprentissage
    #____________
    # paramètres 1ere étape :-> Variation rapide de la temperature
    epochs1 = it; radius_ini1 =Tmax;  radius_fin1 = 1.25;
    etape1=[epochs1,radius_ini1,radius_fin1];
    #
    # paramètres 2ème étape :-> Variation fine de la temperature
    epochs2 = it; radius_ini2 =1.25;  radius_fin2 = Tmin;
    etape2=[epochs2,radius_ini2,radius_fin2];

    sMap.train(etape1=etape1,etape2=etape2, verbose=tracking);
    
    print('Map[%dx%d](%d,%2.2f,%2.2f)(%d,%2.2f,%2.2f) '
      %(nlmap,ncmap,epochs1,radius_ini1,radius_fin1,
         epochs2,radius_ini2,radius_fin2),end='');
    
    return sMap

Xapp = data[108:300,1:5]
sMap=apprentissage(7,7,50,10,0.5)

#etat final
ctk.showmapping(sMap, Xapp, bmus=[], subp=1,seecellid=1)
plt.title("Etat final")

ctk.showmap(sMap)

#erreur topo
bmus1=ctk.mbmus(sMap, Data=Xapp, narg=2)
TE= ctk.errtopo(sMap, bmus2)

#erreur quantification
err = np.mean(getattr(sMap, 'bmu')[1])

print(TE)
print(err)

#affichages des variables sur la carte
ctk.showmap(sMap)

#labeliser 1972 et 1983
bmus2=ctk.mbmus(sMap, Data=Xapp[132-108:144-108], narg=1) #1972
bmus3=ctk.mbmus(sMap, Data=Xapp[264-108:276-108,:], narg=1) #1983