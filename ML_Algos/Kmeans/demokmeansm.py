#-----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import sys
TRIEDPY = "../../../../";       # Chemin d'accès aux modules triedpy
sys.path.append(TRIEDPY);
from   triedpy import triedrdf as rdf
from EX23_enonce import unique
#import random
from   matplotlib import cm
#
TRIEDDATA = "EX23_enonce/";  # Chemin d'accès aux données
#--------------------------------------------------------------
# Some Conditionnements
plt.ion();

# Lecture des données et affichage
Data1 = np.loadtxt(TRIEDDATA+"Data1.mat");
plt.figure;
plt.plot(Data1[:,0], Data1[:,1],'+k');
plt.title('Data as they are');

# Algorithme des k-moyennes
np.random.seed(0);  # (ou pas au choix; pour maitriser l'init des référents)
k = 5;              # Nombre de groupes (i.e. de prototypes)
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
        
formefor=unique.unique(tabforfor)
plt.figure()
#cmap=cm.jet
#Tcol  = cmap(np.arange(1,256,round(256/len(formefor))))
for i in range(len(formefor)):
    plt.plot(Data1[formefor[i],0],Data1[formefor[i],1],"*")#,Tcol[i])
    plt.text(Data1[formefor[i][-1],0],Data1[formefor[i][-1],1],str(len(formefor[i])))
    print("le nombre d'individu de la classe(",i,") = ",len(formefor[i]))

##compteur
for i in range (len(cardi)):
    for j in range (len(classe)):
        if classe[j]==i+1:
            cardi[i]=cardi[i]+1
            
    print("classe ",i+1,"=",cardi[i])