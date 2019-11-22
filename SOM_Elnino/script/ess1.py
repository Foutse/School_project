import numpy as np
from triedpy import triedtools as tls
from triedpy import triedctk   as ctk
from   triedpy import triedsompy as SOM
import matplotlib.pyplot as plt
import seaborn as sns

data_t = np.loadtxt('el_nino.mat')

data = data_t[11*12:23*12,0:5]


def Question_1():
    plt.figure()
    plt.plot(np.arange(len(data)),data[:,1:], '-x')
    plt.legend(['SST1', 'SST2', 'SST3', 'SST4'])
    plt.title("Graphe des données des températures")
    plt.xticks(np.arange(len(data)+12)[::12], range(1972, 1985), rotation=30)
    plt.xlabel("Dates")
    plt.ylabel("Temperatures")
    plt.show()


    ############### Moyenne et écart-type #############
    Moy = np.mean(data[:,1:], axis=0)
    Std = np.std(data[:,1:], axis=0)
    print("Moyenne des 4 variables SSTx : ", Moy)
    print("Écart-type des 4 variables SSTx : ", Std)

    ############### Histogrammes  #############
    plt.subplot(2,2,1)
    plt.hist(data[:,1], bins=45, edgecolor='black')
    plt.title("SST1")
    plt.subplot(2,2,2)
    plt.hist(data[:,2], bins=45, edgecolor='black')
    plt.title("SST2")
    plt.subplot(2,2,3)
    plt.hist(data[:,3], bins=45, edgecolor='black')
    plt.title("SST3")
    plt.subplot(2,2,4)
    plt.hist(data[:,4], bins=45, edgecolor='black')
    plt.title("SST4")

    plt.suptitle("Histogrammes des quatre variables")
    plt.show()
data2 = data_t[108:,0:5]

    ############### Diagrammes de dispersion ###########
    c = np.zeros(len(data[:,1]))
    c[0:11] = np.ones(11)+0.5
    c[len(data[:,1])-11:] = np.ones(11)+1.5
    k=0
    for i in range(1, 5):
        for j in range(i+1, 5):
            k+=1
            plt.subplot(2, 3, k)
            plt.scatter(data[:,i], data[:,j], c=c)
            plt.title("Diag. de disp. SST{0} - SST{1}".format(i, j))
    plt.show()

    corr = np.corrcoef(np.transpose(data[:,1:5]))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(corr, cmap="YlGnBu", mask=mask, square=True, annot=True,
                         xticklabels=['SST1', 'SST2', 'SST3', 'SST4'],
                         yticklabels=['SST1', 'SST2', 'SST3', 'SST4'])
    plt.title("Matrice de corrélations")
    plt.show()

#Question_1()

np.random.seed(1)
sMap  = SOM.SOM('sMap', data[:,1:], mapsize=[6, 6], norm_method='data',
                  initmethod='random', varname=['SST1', 'SST2', 'SST3', 'SST4'])

epochs1 = 20; radius_ini1 =3.00;  radius_fin1 = 1.25
etape1=[epochs1,radius_ini1,radius_fin1]

epochs2 = 50; radius_ini2 =1.25;  radius_fin2 = 0.10
etape2=[epochs2,radius_ini2,radius_fin2]

sMap.train(etape1=etape1,etape2=etape2, verbose='on')

bmus2 = ctk.mbmus(sMap, Data=data[:,1:], narg=2)
TE = ctk.errtopo(sMap, bmus2)
print("\nErreur topographique", TE)

### indice des dates
ind = np.concatenate((range(13), range(11*12, 12*12)))

bmus = ctk.mbmus(sMap, data[:, 1:], narg=1)
ctk.showmapping(sMap, data[:, 1:], bmus, data[:,0], ind, subp=0, seecellid=0)
plt.show()

data_date = data[108:276,0:5]
  ############### Diagrammes de dispersion ###########
c = np.zeros(len(data_date[:,1]))
c[24:36] = np.ones(12)+0.5
c[len(data_date[:,1])-12:] = np.ones(12)+1.5
k=0
for i in range(1, 5):
    for j in range(i+1, 5):
        k+=1
        plt.subplot(2, 3, k)
        plt.scatter(data_date[:,i], data_date[:,j], c=c)
        plt.title(" SST{0} - SST{1}".format(i, j))
plt.suptitle('Diagrame de dispersion des 4 zones 2 à 2 ')
plt.show()


###### régression linéaire minimise les écrts verticaux

#Partie 1 (2)
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
         Dat.append("No_elnino"
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
