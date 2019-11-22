#-----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import sys
TRIEDPY = "../../../../";       # Chemin d'accès aux modules triedpy
sys.path.append(TRIEDPY);
from   triedpy import triedrdf as rdf
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

# Algorithme des k-moyennes
np.random.seed(0);  # (ou pas au choix; pour maitriser l'init des référents)
inert = []
for k in [2, 3, 5, 10, 15, 20]:              # Nombre de groupes (i.e. de prototypes)
    spause = 0.5;       # Controle du temps de pause pour avoir le temps de
#                     voir l'évolution des repositionnements des protos
# L'algo itself 
    protos, classe = rdf.kmoys(Data1,k,spause=spause);
    total = 0
    for i in range(1,k+1):
        ik = np.where(classe==i)[0]
        datak = Data1[ik,:]
        c = datak - protos[i-1,:]
        card = np.size(np.where(classe==i))
        dist=[]
        for j in range(card):
            dist.append(np.dot(c[j,:],c[j,:]))
        inertie = np.round(np.sum(dist)*card/132, 4)
        total += inertie
        print("classe {} :\ncardinal = {}\ninertie={}".format(i,card,inertie))
    print("\n\ninertie intra total : {}".format(total))
    inert.append(total)
#print(k, i, np.size(np.where(classe==i)))


#np.size(np.where(classe==k))
plt.figure()
#y = [0.6182986638, 0.3086724266, 0.2008524098, 0.0860727525, 0.0625540372, 0.0398957516]
x = [2,3,5,10,15,20]
plt.plot(x,inert)
plt.xticks([2,3,5,10,15,20])
plt.title("variation de l'inertie intra pour different k")
plt.xlabel("k")
plt.ylabel("inertie intra")
plt.show()




#-----------------------elements forte-------------------
k = 2
Table = np.zeros((132,25))
for i in range(25):
    protos, classe = rdf.kmoys(Data1,k)
    Table[:,i] = classe
unique, indices, counts = np.unique(Table, return_index = True, return_counts = True, axis=0)


index = []
for i in range(len(indices)):
    tmp=[]
    for j in range(132):
        if list(Table[j,:]==list(unique[i,:])):
            tmp.append(j)
    index.append(tmp) 
plt.figure()
#cmap=cm.jet
#Tcol  = cmap(np.arange(1,256,round(256/len(formefor))))
for i in range(len(index)):
    plt.plot(Data1[index[i],0],Data1[index[i],1],"*")#,Tcol[i])
    plt.text(Data1[index[i][-1],0],Data1[index[i][-1],1],str(len(index[i])))
    print("le nombre d'individu de la classe(",i,") = ",len(index[i]))

##compteur
cardi=np.zeros([k,1])
for i in range (len(cardi)):
    for j in range (len(classe)):
        if classe[j]==i+1:
            cardi[i]=cardi[i]+1
            
    print("classe ",i+1,"=",cardi[i])