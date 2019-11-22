# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 05:15:32 2018
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
TRIEDPY = "../../../../";       # Chemin d'accès aux modules triedpy
sys.path.append(TRIEDPY);
from   triedpy import triedrdf as rdf
#from EX23_enonce import unique
#import random
from   matplotlib import cm
#
TRIEDDATA = "../EX23_enonce/";  # Chemin d'accès aux données
#--------------------------------------------------------------
# Some Conditionnements
plt.ion();

# Lecture des données et affichage
Data1 = np.loadtxt(TRIEDDATA+"Data1.mat");
plt.figure;
plt.plot(Data1[:,0], Data1[:,1],'+k');
plt.title('Data as they are');

def unique(tabforfor):
    indice=[]
    for i in range (132):
            indice.append(i)
        
        #récupération des indices    
    formefor=[]
    for i in range (132):
        if i==indice[0]:
            temp=[]
            for j in indice:
                if list(tabforfor[i,:])==list(tabforfor[j,:]):
                    temp.append(j)
            formefor.append(temp)  
            a=len(temp)
            for k in range(a-1,-1,-1):
                indice.pop(indice.index(temp[k]))
            if len(indice)<1:
                break

    return formefor



# Algorithme des k-moyennes
np.random.seed(0);  # (ou pas au choix; pour maitriser l'init des référents)
k = 3;              # Nombre de groupes (i.e. de prototypes)
spause = 0.5;       # Controle du temps de pause pour avoir le temps de
#                     voir l'évolution des repositionnements des protos
# L'algo itself 
protos, classe = rdf.kmoys(Data1,k)#,spause=spause);
cardi=np.zeros([k,1])
##forme forte
nb=25
tabforfor=np.zeros([132,nb])
tabinit=np.zeros([k,2,nb])
protos, classe = rdf.kmoys(Data1,k)


for i in range (nb):
    protos, classe = rdf.kmoys(Data1,k)
    tabforfor[:,i]=classe
    tabinit[:,:,i]=protos
    
indice=[]
for i in range (132):
        indice.append(i)
        
formefor=unique(tabforfor)
plt.figure()
#cmap=cm.jet
#Tcol  = cmap(np.arange(1,256,round(256/len(formefor))))
for i in range(len(formefor)):
    plt.plot(Data1[formefor[i],0],Data1[formefor[i],1],"*")#,Tcol[i])
    plt.text(Data1[formefor[i][-1],0],Data1[formefor[i][-1],1],str(len(formefor[i])))
    plt.title("Kmeans algorithme on data with k=%d Over 25 trials" % (k))
    plt.xlabel("Representation des formes fortes et leurs cardinalite")
    print("le nombre d'individu de la classe(",i,") = ",len(formefor[i]))

##compteur
for i in range (len(cardi)):
    for j in range (len(classe)):
        if classe[j]==i+1:
            cardi[i]=cardi[i]+1
            
    print("classe ",i+1,"=",cardi[i])




