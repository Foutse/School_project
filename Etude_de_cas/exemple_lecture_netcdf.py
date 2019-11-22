
# coding: utf-8

# ### Lecture des donnees pour etude de cas :
# ## _Etude de la variabilité du phytoplancton au niveau de la Mer Méditerranée en utilisant les cartes auto-organisatrices (SOM)_

# In[1]:


# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import glob
import netCDF4
from mpl_toolkits.basemap import Basemap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ### _Exemple de lecture des données `Chl` :_

# In[2]:


# repertoires de donnees pour chaque parametre
datadir1 ='./data/PFT'
datadir2 ='./data/SST'
datadir3 ='./data/Chl' #???
datadir4 ='./data/412'
datadir5 ='./data/443'
datadir6 ='./data/490'
datadir7 ='./data/555'
#datadir1 =  r'D:\docTried\etude_de_cas\data\PFT' #r'C:\Users\Lucas\Documents\Etude de cas\data\PFT'
#datadir2 =  r'D:\docTried\etude_de_cas\data\SST' #C:\Users\Lucas\Documents\Etude de cas\data\SST'
#datadir3 =  r'D:\docTried\etude_de_cas\data\Chl'
#datadir4 =  r'D:\docTried\etude_de_cas\data\412'
#datadir5 =  r'D:\docTried\etude_de_cas\data\443'
#datadir6 =  r'D:\docTried\etude_de_cas\data\490'
#datadir7 =  r'D:\docTried\etude_de_cas\data\555'
#6 variables satellitaires (Chla(OC5), 4 Rrs(λ) et SST)
# In[3]:


# listes de fichiers 
filelist1 = glob.glob(datadir1 + os.sep + "*.nc")
filelist2 = glob.glob(datadir2 + os.sep + "*.nc")
filelist3 = glob.glob(datadir3 + os.sep + "*.nc")
filelist4 = glob.glob(datadir4 + os.sep + "*.nc")
filelist5 = glob.glob(datadir5 + os.sep + "*.nc")
filelist6 = glob.glob(datadir6 + os.sep + "*.nc")
filelist7 = glob.glob(datadir7 + os.sep + "*.nc")


# In[4]:


datadir1 + os.sep + "*.nc"
datadir2 + os.sep + "*.nc"
datadir3 + os.sep + "*.nc"
datadir4 + os.sep + "*.nc"
datadir5 + os.sep + "*.nc"
datadir6 + os.sep + "*.nc"
datadir7 + os.sep + "*.nc"


# In[5]:


#filelist1


# In[6]:


# Lecture d'un fichier NetCDF ...
ific = 0


# ### lecture  ... à mettre dans une boucle pour lire tous les données Chl  ...
# 
# ### idem pour les autres variables  ...
# 

# In[7]:


# lecture de l'element ific, ...a mettre dans un boucle pour lire tous les autres ...
ncfile = filelist1[ific]
len(filelist1)
arraylist = []
for i in range(len(filelist1)):
    arraylist.append(netCDF4.Dataset(filelist1[i]).variables)
    

matrixchla = np.zeros((2,3,10))

#second_col = matrix[:,1,:]
# In[8]:
#
#
#ficnom = os.path.basename(ncfile) # recupere le nom du fichier, sans le chemin
#
## recupere la date a partir du nom (encadree entre le premier et deuxieme tiret bas '_')
#i0 = ficnom.find('_')  # cherche la position du premier '_', 
#i1 = ficnom.find('_',i0+1)  # cherche la position du deuxieme '_'
#j0 = ficnom[i0:i1].find('-')  # cherche dans l'intervalle la position du premier '-', 
#
## recupere les caracteres qui devraient correspondre a la date de depart
#date_ini = int(ficnom[(i0+1):(i0+j0)]);
#
## recupere les caracteres qui devraient correspondre a la date de fin
#date_fin = int(ficnom[(i0+j0+1):(i1)]);
#
## data ini et fin au format numerique, a vous de la separer
## en annee, mois et jour, si vous en avez besoin
#print(date_ini,date_fin)
#

# In[9]:


# Lecture d'un fichier NetCDF ...
nc = netCDF4.Dataset(ncfile);
liste_var = nc.variables;       # mois par mois de janvier 1930 à decembre 1960 I guess

# Prospection pour connaitre le nom des variables contenues dans le fichier
# a faire manuellement pour les autres donnees: PFT, SST, 412, ...
#print("\n Fichier NetCDF:\n    {}\n\n Variables trouvées dans ce fichier:".format(ficnom))
for ivar,var in enumerate(liste_var.keys()) :
    print("    var {:d} ... ''{:s}''".format(ivar,var))


# In[10]:


# Extraction des variables d'un fichier NetCDF
# les coordonnees
#lon = liste_var['lon']
#lat = liste_var['lat']
##convert from masked to array
#lon = np.ma.filled(lon, np.nan)
#lat = np.ma.filled(lat, np.nan)
#
##dimensions
#nlon = lon.shape[0]
#nlat = lat.shape[0]

nlat=433
nlon=769

data1 = np.ndarray((184,7,nlat,nlon), dtype=float)

for i in range(1):
    ncfile1 = filelist1[i]
    nc1 = netCDF4.Dataset(ncfile1);
    liste_var1 = nc1.variables;       # mois par mois de janvier 1930 à decembre 1960 I guess

    ncfile2 = filelist2[i]
    nc2 = netCDF4.Dataset(ncfile2);
    liste_var2 = nc2.variables;       # mois par mois de janvier 1930 à decembre 1960 I guess

    ncfile3 = filelist3[i]
    nc3 = netCDF4.Dataset(ncfile3);
    liste_var3 = nc3.variables;       # mois par mois de janvier 1930 à decembre 1960 I guess

    ncfile4 = filelist4[i]
    nc4 = netCDF4.Dataset(ncfile4);
    liste_var4 = nc4.variables;       # mois par mois de janvier 1930 à decembre 1960 I guess

    ncfile5 = filelist5[i]
    nc5 = netCDF4.Dataset(ncfile5);
    liste_var5 = nc5.variables;       # mois par mois de janvier 1930 à decembre 1960 I guess

    ncfile6 = filelist6[i]
    nc6 = netCDF4.Dataset(ncfile6);
    liste_var6 = nc6.variables;       # mois par mois de janvier 1930 à decembre 1960 I guess
    
    ncfile = filelist7[i]
    nc = netCDF4.Dataset(ncfile);
    liste_var7 = nc.variables;       # mois par mois de janvier 1930 à decembre 1960 I guess
    
    pft_tmp = liste_var1['PFT']
    sst_tmp = liste_var2['SST']
    chl_tmp = liste_var3['CHL-OC5_mean']
    nrrs412_tmp = liste_var4['NRRS412_mean']
    nrrs443_tmp = liste_var5['NRRS443_mean']
    nrrs490_tmp = liste_var6['NRRS490_mean']
    nrrs555_tmp = liste_var7['NRRS555_mean']
    
    #converting -999.0 values to nan##########################################
    chl_tmp = np.ma.filled(chl_tmp, np.nan)
    chl_tmp[chl_tmp == -999.0] = np.nan

    sst_tmp = np.ma.filled(sst_tmp, np.nan)
    sst_tmp[sst_tmp == -999.0] = np.nan

    pft_tmp = np.ma.filled(pft_tmp, np.nan)
    pft_tmp[pft_tmp == -999.0] = np.nan

    nrrs412_tmp = np.ma.filled(nrrs412_tmp, np.nan)
    nrrs412_tmp[nrrs412_tmp == -999.0] = np.nan

    nrrs443_tmp = np.ma.filled(nrrs443_tmp, np.nan)
    nrrs443_tmp[nrrs443_tmp == -999.0] = np.nan

    nrrs490_tmp = np.ma.filled(nrrs490_tmp, np.nan)
    nrrs490_tmp[nrrs490_tmp == -999.0] = np.nan

    nrrs555_tmp = np.ma.filled(nrrs555_tmp, np.nan)
    nrrs555_tmp[nrrs555_tmp == -999.0] = np.nan
        
    data1[i,0]= np.nanmean(pft_tmp)
    data1[i,1]= np.nanmean(sst_tmp)
    data1[i,2]= np.nanmean(chl_tmp)
    data1[i,3]= np.nanmean(nrrs412_tmp)
    data1[i,4]= np.nanmean(nrrs443_tmp)
    data1[i,5]= np.nanmean(nrrs490_tmp)
    data1[i,6]= np.nanmean(nrrs555_tmp)



data = np.reshape(data1, (184, 7, nlon*nlat))

cols = ["NRRS555_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean","PFT","SST","CHL-OC5_mean"]

datasd = pd.DataFrame(data, columns = cols)

np.save("Tous_Donnees", data)

x = data[0,:,:]
dt =x[ ~np.any(np.isnan(x), axis=0)]

dataR = data[0,:,:] #On prend les donnees d'une semaines



summ= np.sum(dataR,axis = 0)
ind = np.where(~np.isnan(summ))
dataR = dataR[:,ind]
dataR  = np.reshape(dataR, (7, len(ind[0]))) #np.reshape(ind)[1]))
cols = ["NRRS555_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean","PFT","SST","CHL-OC5_mean"]

datasd = pd.DataFrame(dataR.T, columns = cols)

cor = datasd.corr()
#plt.matshow(cor)
#plt.
plt.figure()
corr = np.corrcoef(np.transpose(datasd))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
        ax = sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), mask=mask, square=True, annot=True,
                         xticklabels=["NRRS555","NRRS490","NRRS443","NRRS412","PFT","SST","CHL-OC5"],
                         yticklabels=["NRRS555","NRRS490","NRRS443","NRRS412","PFT","SST","CHL-OC5"])
plt.title("Matrice de corrélations")
plt.show()



dataTrans = StandardScaler().fit_transform(datasd)
dataTrans = pd.DataFrame(dataTrans, columns=cols)

cols1 = ["NRRS555","NRRS490","NRRS443","NRRS412","PFT","SST","CHL-OC5"]
pca = PCA(svd_solver='full')
acpdata = pca.fit_transform(dataTrans.T)
fig, ax = plt.subplots()
ax.set_xlim(-400,400)
ax.set_ylim(-400,400)
plt.scatter(acpdata[:,0], acpdata[:,1])
for i in range(7):
    plt.annotate(cols1[i],(acpdata[i,0], acpdata[i,1]))

print(pca.n_components_)
print(pca.explained_variance_)