# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 01:23:13 2018

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import cm
from   triedpy import triedtools as tls
import pandas as pd
import seaborn as sns
from   triedpy import triedctk   as ctk
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
#
if 1 :
    TRIEDPY = "../../../../";   # Mettre le chemin d'accès ...
    #TRIEDPY = "C:/Users/Charles/Documents/FAD/FAD_Charles";
    #TRIEDPY = "C:/Users/Charles/Documents/FAD/FAD_Charles/WORKZONE/Python3";
    TRIEDPY = "../../../../";  # Mettre le chemin d'accès ... 9
    sys.path.append(TRIEDPY);  # ... aux modules triedpy
#from   triedpy import triedtools as tls
from   triedpy import triedctk   as ctk

import TPC04_methodes as TPC04_methodes

data = np.loadtxt("el_nino.mat");

#x = data[108:436:8,0] #/1000

t = 1/12
mois = np.array([0.0, t, 2*t, 3*t, 4*t, 5*t, 6*t, 7*t, 8*t, 9*t, 10*t, 11*t])
ann=[]
for i in range(70,98):
    if i ==97:
        for j in range(4):
            ann.append(i + mois[j])
    else:
        for j in range(12):
            ann.append(i + mois[j])
            
dataApp = data[108:300,1:5]
dateApp = data[108:300,0]
dataTest = data[301:436,1:5]
dateTest =  data[301:436,0]
x = ann
plt.figure()
plt.subplot(2,2,1)
plt.plot(x, data[108:436,1],'r-',markersize=8);
plt.grid(1)
plt.xlabel("Mean= %f, Ecart-type = %f" %(np.mean(data[108:436,1]), np.sqrt(np.var(data[108:436,1]))))
plt.title("Région 1 ", fontsize = 8)
plt.subplot(2,2,2)
plt.plot(x, data[108:436,2],'g-',markersize=8);
plt.grid(1)
plt.xlabel("Mean= %f, Ecart-type = %f" %(np.mean(data[108:436,2]),np.sqrt(np.var(data[108:436,2]))) )
plt.title(" Région 2 ", fontsize = 8)
plt.subplot(2,2,3)
plt.plot(x, data[108:436,3],'b-',markersize=8);
plt.grid(1)
plt.xlabel("Mean= %f, Ecart-type = %f" %(np.mean(data[108:436,3]),np.sqrt(np.var(data[108:436,3]))) )
plt.title("Région 3 ", fontsize = 8)
plt.subplot(2,2,4)
plt.plot(x, data[108:436,4],'k-',markersize=8);
plt.grid(1)
plt.xlabel("Mean= %f, Ecart-type = %f" %(np.mean(data[108:436,4]), np.sqrt(np.var(data[108:436,4]))) )
plt.title("Région 4", fontsize = 8)


plt.figure()
plt.subplot(2,2,1)
plt.hist( data[108:436,1]);
plt.xlabel("Min = %f, Max= %f" %(np.min(data[108:436,1]), np.max(data[108:436,1])));
plt.title("Température moyenne région 1 ", fontsize = 8)
plt.subplot(2,2,2)
plt.hist( data[108:436,2]);
plt.xlabel("Min = %f, Max= %f" %(np.min(data[108:436,2]), np.max(data[108:436,2])));
plt.title(" Température moyenne région 2 ", fontsize = 8)
plt.subplot(2,2,3)
plt.hist( data[108:436,3]);
plt.xlabel("Min = %f, Max= %f" %(np.min(data[108:436,3]), np.max(data[108:436,3])));
plt.title("Température moyenne région 3 ", fontsize = 8)
plt.subplot(2,2,4)
plt.hist( data[108:436,4]);
plt.xlabel("Min = %f, Max= %f" %(np.min(data[108:436,4]), np.max(data[108:436,4])));
plt.title("Température moyenne région 4", fontsize = 8)

c = np.zeros(len(data[108:436,1:5]))
c[0:11] = np.ones(11)+0.5
c[len(data[:,1])-11:] = np.ones(11)+1.5
#k=0
plt.figure()
plt.subplot(2,3,1)
plt.scatter(data[108:436,1], data[108:436,2],c=c, marker='*');
plt.title("Température moyenne région 1 & 2 ", fontsize = 8)
plt.subplot(2,3,2)
plt.scatter(data[108:436,2], data[108:436,3],c=c, marker='*');
plt.title("Température moyenne région 2 & 3 ", fontsize = 8)
plt.subplot(2,3,3)
plt.scatter( data[108:436,3], data[108:436,4],c=c, marker='*');
plt.title("Température moyenne région 3 & 4 ", fontsize = 8)
plt.subplot(2,3,4)
plt.scatter( data[108:436,4], data[108:436,1],c=c, marker='*');
plt.title("Température moyenne région 4 & 1", fontsize = 8)
plt.subplot(2,3,5)
plt.scatter(data[108:436,1], data[108:436,3],c=c, marker='*');
plt.title("Température moyenne région 1 & 3 ", fontsize = 8)
plt.subplot(2,3,6)
plt.scatter(data[108:436,2], data[108:436,4],c=c, marker='*');
plt.title("Température moyenne région 2 & 4 ", fontsize = 8)

#La correlation
f, ax = plt.subplots(figsize=(10, 8))
#corr = dataframe.corr()
dat= pd.DataFrame(data[108:436,1:5])
donn =dat.corr()
sns.heatmap(donn, mask=np.zeros_like(donn, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

#np.corrcoef(data[108:436,1:5])
#donn= pd.DataFrame(data[108:436,1:5])
#plt.matshow(donn.corr())
#plt.matshow(np.corrcoef(data[108:436,1:4]))

#2) Etude par carte topologique :




def app_donnees(donnee, classname) :
    '''
    % Code pour l'apprentissage en 2 phases de la Carte
    % Topologique
    En sortie :
    sMap       : La structure de la carte
    Xapp, Xapplabels : L'ensemble d'apprentissage et le labels (classes) associés
    classnames : Nom des classes (qui on servis à la labelisation)
    '''
    from   triedpy import triedsompy as SOM
    
#    if 0 : # Affichage des seules données
#        plt.figure();
#        plt.plot(donnee);
    #==================================================================
    # la CARTE TOPOLOGIQUE
    #------------------------------------------------------------------
    # Choix des dimensions de la CT : remplacer les 0  par les valeurs
    # souhaitees des nombres de lignes (nlmap) et de colonne (ncmap)
    nlmap = 7;  ncmap = 7; # Nombre de lignes et nombre de colones
    if nlmap<=0 or ncmap<=0 :
        print("app_lettre : mauvais choix de dimension pour la carte");
        sys.exit(0);
    #
    # Creation d'une structure de carte initialisee (référents non initialisés)
    initmethod='random'; # 'random', 'pca'
    sMap  = SOM.SOM('sMap', donnee, mapsize=[nlmap, ncmap], norm_method='data', \
                  initmethod=initmethod, varname=classnames)
    #==================================================================
    # APPRENTISSAGE 
    #------------------------------------------------------------------
    tracking = 'on';  # niveau de suivi de l'apprentissage
    #____________
    # paramètres 1ere étape :-> Variation rapide de la temperature
    epochs1 = 20; radius_ini1 =5.00;  radius_fin1 = 1.25;
    etape1=[epochs1,radius_ini1,radius_fin1];
    #
    # paramètres 2ème étape :-> Variation fine de la temperature
    epochs2 = 50; radius_ini2 =1.25;  radius_fin2 = 0.10;
    etape2=[epochs2,radius_ini2,radius_fin2];
    #
    # Avec Sompy, les paramètres des 2 étapes sont passés en même temps pour l'
    # apprentissage de la carte.
    sMap.train(etape1=etape1,etape2=etape2, verbose=tracking);
    #
    print('Map[%dx%d](%d,%2.2f,%2.2f)(%d,%2.2f,%2.2f) '
      %(nlmap,ncmap,epochs1,radius_ini1,radius_fin1,
         epochs2,radius_ini2,radius_fin2),end='');
    ctk.showmapping(sMap, donnee, bmus=[], seecellid=1, subp=True,override=False);
    plt.title("Etat Final");
    #tmp = ctk.classifperf(sMap, Xapp, Xapplabels)
    return sMap, donnee, classnames


classnames = ["SST1","SST2","SST3","SST4"]
#data_70_83 = 
sMap, donnee, classenames = app_donnees(data[108:436,1:5], classnames)

ctk.showmap(sMap)


ctk.showcarte(sMap)
###############################################################################
##donnee et annee                                                             #
annee = data[108:436,0]
data1 = data[108:436,1:5]
data_73_82 =np.concatenate((annee[36:48,],annee[144:156,]), axis=0)
donnees_73_82 = np.concatenate((data1[36:48,],data1[144:156,]), axis=0)

bmus1 = ctk.mbmus(sMap,Data=donnees_73_82);

data_73_82= [int(i) for i in data_73_82]

ctk.showmap(sMap,sztext=5, nodes=bmus1, Labels=[str(i) for i in data_73_82]); #,dh=-0.04, dv=-0.04);
###############################################################################

#2.2) Classification ascendante hiérarchique (CAH) des neurones de la carte
referent = getattr(sMap, 'codebook')
plt.figure()
Z = linkage(referent, 'ward')
n = dendrogram(Z)
classe = fcluster(Z, 3, criterion='maxclust')




bmus3 = ctk.mbmus(sMap, Data=data[108:436,1:5]);
data1 = []
for j in range(328):
    data1.append(classe[bmus3[j]][0])
           

s=np.reshape([str(i) for i in data1],(len(data1),1))
Tfreq,Ulab = ctk.reflabfreq(sMap,data[108:436,1:5],s);
CBlabmaj = ctk.cblabvmaj(Tfreq,Ulab);
CBilabmaj = ctk.label2ind(CBlabmaj, s); # transformation des labels en int
ctk.showcarte(sMap,figlarg=12, fighaut=12,shape='s',shapescale=600,\
    colcell=CBilabmaj,text=CBlabmaj,\
    sztext=16,cmap=cm.jet,showcellid=False);



#on le classe en 2 classe

referent = getattr(sMap, 'codebook')
plt.figure()
Z = linkage(referent, 'ward')
n = dendrogram(Z)
classe = fcluster(Z, 2, criterion='maxclust')




bmus2 = ctk.mbmus(sMap, Data=data[108:436,1:5]);
data1 = []
for j in range(328):
    data1.append(classe[bmus2[j]][0])
           

s=np.reshape([str(i) for i in data1],(len(s),1))
Tfreq,Ulab = ctk.reflabfreq(sMap,data[108:436,1:5],s);
CBlabmaj = ctk.cblabvmaj(Tfreq,Ulab);
CBilabmaj = ctk.label2ind(CBlabmaj, s); # transformation des labels en int
ctk.showcarte(sMap,figlarg=12, fighaut=12,shape='s',shapescale=600,\
    colcell=CBilabmaj,text=CBlabmaj,\
    sztext=16,cmap=cm.jet,showcellid=False);



ctk.showbarcell(sMap)

#Hypothese géophysiciens
sc= StandardScaler()

NormData = sc.fit_transform(data[108:436,1:5])


normalized_data = (data[108:436,1] - np.mean(data[108:436,1]))/(np.sqrt(np.var(data[108:436,1])))


normalized_data = normalized_data.T
Dat = []
for i in range(328):
     if normalized_data[i] > 1:
         Dat.append("elnino")
     else:
         Dat.append("No_elnino")
         


#on le classe en 2 classe

s=np.reshape([str(i) for i in Dat],(len(Dat),1))
Tfreq,Ulab = ctk.reflabfreq(sMap,data[108:436,1:5],s);
CBlabmaj = ctk.cblabvmaj(Tfreq,Ulab);
CBilabmaj = ctk.label2ind(CBlabmaj, s); # transformation des labels en int
ctk.showcarte(sMap,figlarg=12, fighaut=12,shape='s',shapescale=600,\
    colcell=CBilabmaj,text=CBlabmaj,\
    sztext=8,cmap=cm.jet,showcellid=False);

           
#classer sur la carte

Donnee = data[108:436,0:5]           
indice = np.where(NormData[:,0]>1)
dateElnino = Donnee[indice[0],0]
donneesElnino = Donnee[indice[0],1:5]

bmusEl = ctk.mbmus(sMap,Data=donneesElnino);

data_73_82= [int(i) for i in dateElnino]

ctk.showmap(sMap,sztext=5, nodes=bmusEl, Labels=[str(i) for i in dateElnino])