# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 15:59:20 2019
"""

from pylab import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd

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
    TRIEDPY = "../../../../";  # Mettre le chemin d'accès ... 9
    sys.path.append(TRIEDPY);  # ... aux modules triedpy
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
            
temps = ann
dataAll = data[108:436:, 1:5]
dataApp = data[108:300,1:5]
dateApp = data[108:300,0]
dataTest = data[301:436,1:5]
dateTest =  data[301:436,0]

#graphes des données
plt.figure()
ax = plt.subplot(111)
plt.plot(temps,dataAll[:,0],label='Zone 1')
plt.plot(temps,dataAll[:,1],label='Zone 2')
plt.plot(temps,dataAll[:,2],label='Zone 3')
plt.plot(temps,dataAll[:,3],label='Zone 4')
plt.legend()
plt.grid(1)
plt.xlabel('Temps(Les données vont du mois de janvier 1970 à avril 1997)')
plt.ylabel('Température Moyenne')
plt.title('Variation de la température moyenne au fil des années dans les 4 zones d études')
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.xaxis.grid(True,'minor')
ax.yaxis.grid(True,'minor')
ax.xaxis.grid(True,'major',linewidth=2)
ax.yaxis.grid(True,'major',linewidth=1)



plt.figure()
plt.subplot(2,2,1)
plt.hist( dataAll[:,0], bins=45, edgecolor='black');
plt.xlabel("Min =%0.2f, Max= %0.2f, Moy = %0.2f, Etp = %0.2f" %(np.min(dataAll[:,0]), np.max(dataAll[:,0]),np.mean(dataAll[:,0]), np.sqrt(np.var(dataAll[:,0]))));
plt.title(" Zone 1 ", fontsize = 11)
plt.subplot(2,2,2)
plt.hist( dataAll[:,1], bins=45, edgecolor='black');
plt.xlabel("Min =%0.2f, Max= %0.2f, Moy = %0.2f, Etp = %0.2f" %(np.min(dataAll[:,1]), np.max(dataAll[:,1]),np.mean(dataAll[:,1]), np.sqrt(np.var(dataAll[:,1]))));
plt.title(" Zone 2 ", fontsize = 11)
plt.subplot(2,2,3)
plt.hist( dataAll[:,2], bins=45, edgecolor='black');
plt.xlabel("Min =%0.2f, Max= %0.2f, Moy = %0.2f, Etp = %0.2f" %(np.min(dataAll[:,2]), np.max(dataAll[:,2]),np.mean(dataAll[:,2]), np.sqrt(np.var(dataAll[:,2]))));
plt.title("Zone 3 ", fontsize = 11)
plt.subplot(2,2,4)
plt.hist( dataAll[:,3], bins=45, edgecolor='black');
plt.xlabel("Min =%0.2f, Max= %0.2f, Moy = %0.2f, Etp = %0.2f" %(np.min(dataAll[:,3]), np.max(dataAll[:,3]), np.mean(dataAll[:,3]), np.sqrt(np.var(dataAll[:,3]))));
plt.title(" Zone 4", fontsize = 11)
plt.suptitle('Température moyenne des 4 Zones ')

dataApp = data[108:300,1:5]
annee = data[108:436,0]
data1 = data[108:436,1:5]
data_73_82 =np.concatenate((annee[24:36,],annee[156:168,]), axis=0)
donnees_73_82 = np.concatenate((data1[24:36,],data1[156:168,]), axis=0)

data_date = data[108:,0:5]
  ############### Diagrammes de dispersion ###########
c = np.zeros(len(data_date[:,1]))
c[24:36] = np.ones(12)+0.5
c[len(data_date[:,1])-172:168] = np.ones(12)+1.5
k=0
plt.figure()
for i in range(1, 5):
    for j in range(i+1, 5):
        k+=1
        plt.subplot(2, 3, k)
        plt.scatter(data_date[:,i], data_date[:,j], c=c,marker='*')
        plt.title(" SST{0} - SST{1}".format(i, j))
plt.suptitle('Diagrame de dispersion des 4 zones 2 à 2 ')
plt.show()


plt.figure()
corr = np.corrcoef(np.transpose(data_date[:,1:5]))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
        ax = sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), mask=mask, square=True, annot=True,
                         xticklabels=['SST1', 'SST2', 'SST3', 'SST4'],
                         yticklabels=['SST1', 'SST2', 'SST3', 'SST4'])
plt.title("Matrice de corrélations")
plt.show()

f, ax = plt.subplots(figsize=(10, 8))
#corr = dataframe.corr()
dat= pd.DataFrame(data[108:436,1:5])
donn =dat.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(donn, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True),annot=True,
            square=True,xticklabels=['SST1', 'SST2', 'SST3', 'SST4'],yticklabels=['SST1', 'SST2', 'SST3', 'SST4'], ax=ax)
plt.title("Matrice de corrélations")

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
                  initmethod=initmethod, varname=classname)
    #==================================================================
    # APPRENTISSAGE 
    #------------------------------------------------------------------
    tracking = 'on';  # niveau de suivi de l'apprentissage
    #____________
    # paramètres 1ere étape :-> Variation rapide de la temperature
    epochs1 = 50; radius_ini1 =10;  radius_fin1 = 1.25;
    etape1=[epochs1,radius_ini1,radius_fin1];
    #
    # paramètres 2ème étape :-> Variation fine de la temperature
    epochs2 = 50; radius_ini2 =10;  radius_fin2 = 0.50;
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
    plt.suptitle('Etat final ')
    #tmp = ctk.classifperf(sMap, Xapp, Xapplabels)
    return sMap, donnee, classname

dataApp = data[108:300,1:5]
dateApp = data[108:300,0]
dataTest = data[300:436,1:5]
dateTest =  data[300:436,0]
np.seed(0)
classname = ["SST1","SST2","SST3","SST4"]
#data_70_83 = 
sMap, donnee, classename = app_donnees(dataApp, classname)

ctk.showmap(sMap)



#annee = data[108:436,0]
#data1 = data[108:436,1:5]
data_72_83 =np.concatenate((dateApp[24:36,],dateApp[156:168,]), axis=0)
donnees_72_83 = np.concatenate((dataApp[24:36,],dataApp[156:168,]), axis=0)

#data_73_82 =np.concatenate((annee[36:48,],annee[144:156,]), axis=0)
#donnees_73_82 = np.concatenate((data1[36:48,],data1[144:156,]), axis=0)

bmus1 = ctk.mbmus(sMap,Data=donnees_72_83);

data_72_83= [int(i) for i in data_72_83]

ctk.showmap(sMap,sztext=5, nodes=bmus1, Labels=[str(i) for i in data_72_83]); #,dh=-0.04, dv=-0.04);
#######

#2.2) Classification ascendante hiérarchique (CAH) des neurones de la carte
referent = getattr(sMap, 'codebook')
plt.figure()
Z = linkage(referent, 'ward')
n = dendrogram(Z)
classe = fcluster(Z, 3, criterion='maxclust')




bmus3 = ctk.mbmus(sMap, Data=dataApp);
data1 = []
for j in range(192):
    data1.append(classe[bmus3[j]][0])
    
s=np.reshape([str(i) for i in data1],(len(data1),1))
Tfreq,Ulab = ctk.reflabfreq(sMap,dataApp,s);
CBlabmaj = ctk.cblabvmaj(Tfreq,Ulab);
CBilabmaj = ctk.label2ind(CBlabmaj, s); # transformation des labels en int
ctk.showcarte(sMap,figlarg=8, fighaut=8,shape='h',shapescale=400,\
    colcell=CBilabmaj,text=CBlabmaj,\
    sztext=16,cmap=cm.jet,showcellid=False);

#on le classe en 2 classe

referent = getattr(sMap, 'codebook')
plt.figure()
Z = linkage(referent, 'ward')
n = dendrogram(Z)
classe2 = fcluster(Z, 2, criterion='maxclust')




bmus2 = ctk.mbmus(sMap, Data=dataApp);
data1 = []
for j in range(192):
    data1.append(classe2[bmus2[j]][0])
           

s=np.reshape([str(i) for i in data1],(len(s),1))
Tfreq,Ulab = ctk.reflabfreq(sMap,dataApp,s);
CBlabmaj = ctk.cblabvmaj(Tfreq,Ulab);
CBilabmaj = ctk.label2ind(CBlabmaj, s); # transformation des labels en int
ctk.showcarte(sMap,figlarg=8, fighaut=8,shape='h',shapescale=400,\
    colcell=CBilabmaj,text=CBlabmaj,\
    sztext=16,cmap=cm.jet,showcellid=False);

#Hypothese géophysiciens
sc= StandardScaler()


NormData = sc.fit_transform(dataApp)


normalized_data = (dataApp[:,1] - np.mean(dataApp[:,1]))/(np.sqrt(np.var(dataApp[:,1])))


normalized_data = normalized_data.T
Dat = []
for i in range(192):
     if normalized_data[i] > 1:
         Dat.append("elnino")
     else:
         Dat.append("No_elnino")
         


#on le classe en 2 classe

s=np.reshape([str(i) for i in Dat],(len(Dat),1))
Tfreq,Ulab = ctk.reflabfreq(sMap,dataApp,s);
CBlabmaj = ctk.cblabvmaj(Tfreq,Ulab);
CBilabmaj = ctk.label2ind(CBlabmaj, s); # transformation des labels en int
ctk.showcarte(sMap,figlarg=12, fighaut=12,shape='s',shapescale=600,\
    colcell=CBilabmaj,text=CBlabmaj,\
    sztext=8,cmap=cm.jet,showcellid=False);
              
              
#classer sur la carte
#dataApp1 = data[108:300,0:5]

#Donnee = dataApp1           
indice = np.where(NormData[:,0]>1)
dateElnino = dateApp[indice[0]]
donneesElnino = dataApp[indice[0]]

bmusEl = ctk.mbmus(sMap,Data=donneesElnino);

data_72_83= [int(i) for i in dateElnino]

ctk.showmap(sMap,sztext=5, nodes=bmusEl, Labels=[str(i) for i in data_72_83], dh=0.03, dv=-0.04)

#Prediction
pred_bmus = sMap.project_data(dataTest)
pred_label = classe2[pred_bmus[:]]
pred_label = np.reshape([str(i) for i in pred_label], (len(pred_label),1))

Tfreq,Ulab = ctk.reflabfreq(sMap,dataTest,pred_label);
CBlabmaj = ctk.cblabvmaj(Tfreq,Ulab);
CBilabmaj = ctk.label2ind(CBlabmaj, pred_label); # transformation des labels en int
ctk.showcarte(sMap,hits = None, figlarg=10, fighaut=10,shape='h',shapescale=300,\
    colcell=CBilabmaj,text=CBlabmaj,dv=0.02,dh=0.0,\
    sztext=8,cmap=cm.jet,showcellid=True);