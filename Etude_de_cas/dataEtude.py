# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 17:34:51 2019

@author: FOUTSE
"""

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
from sklearn.preprocessing import Imputer #as imp
#sklearn.preprocessing.Imputer.fit_transform
#path = ''
donne_ann1 = np.load("datanonan1.npy")
donne_ann2 = np.load("datanonan2.npy")
donne_ann3 = np.load("datanonan3.npy")
donne_ann4 = np.load("datanonan4.npy")

cols = ["CHL-OC5", "SST","NRRS412","NRRS443","NRRS490", "NRRS555"]
datasd = pd.DataFrame(np.delete(donne_ann1.T,2, axis = 1), columns = cols)

#dataA = datasd
cor = datasd.corr()
#plt.matshow(cor)
#plt.
plt.figure()
corr = np.corrcoef(np.transpose(datasd))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
        ax = sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), mask=mask, square=True, annot=True,
                         xticklabels=["CHL-OC5", "SST","NRRS412","NRRS443","NRRS490", "NRRS555"],
                         yticklabels=["CHL-OC5", "SST","NRRS412","NRRS443","NRRS490", "NRRS555"])
plt.title("Matrice de corrélations Premiere annee")
plt.show()



datasd = pd.DataFrame(np.delete(donne_ann2.T,2, axis = 1), columns = cols)

#dataA = datasd
cor = datasd.corr()
#plt.matshow(cor)
#plt.
plt.figure()
corr = np.corrcoef(np.transpose(datasd))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
        ax = sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), mask=mask, square=True, annot=True,
                         xticklabels=["CHL-OC5", "SST","NRRS412","NRRS443","NRRS490", "NRRS555"],
                         yticklabels=["CHL-OC5", "SST","NRRS412","NRRS443","NRRS490", "NRRS555"])
plt.title("Matrice de corrélations deuxieme annee")
plt.show()



datasd = pd.DataFrame(np.delete(donne_ann3.T,2, axis = 1), columns = cols)

#dataA = datasd
cor = datasd.corr()
#plt.matshow(cor)
#plt.
plt.figure()
corr = np.corrcoef(np.transpose(datasd))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
        ax = sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), mask=mask, square=True, annot=True,
                         xticklabels=["CHL-OC5", "SST","NRRS412","NRRS443","NRRS490", "NRRS555"],
                         yticklabels=["CHL-OC5", "SST","NRRS412","NRRS443","NRRS490", "NRRS555"])
plt.title("Matrice de corrélations Troisieme annee")
plt.show()

datasd = pd.DataFrame(np.delete(donne_ann4.T,2, axis = 1), columns = cols)

#dataA = datasd
cor = datasd.corr()
#plt.matshow(cor)
#plt.
plt.figure()
corr = np.corrcoef(np.transpose(datasd))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
        ax = sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), mask=mask, square=True, annot=True,
                         xticklabels=["CHL-OC5", "SST","NRRS412","NRRS443","NRRS490", "NRRS555"],
                         yticklabels=["CHL-OC5", "SST","NRRS412","NRRS443","NRRS490", "NRRS555"])
plt.title("Matrice de corrélations Quatrieme annee")
plt.show()


dataAll = np.load("donneeSurMernoNan.npy")
dataAllwithnana = np.load("donneeSurMer.npy")

pft = dataAllwithnana[:,2,:]
pft = pd.DataFrame(pft)
pft = pft.fillna(0)
np.save("pftdata", pft)
chl_tmp = dataAll[:,0,:]
sst_tmp = dataAll[:,1,:]
nrrs412_tmp = dataAll[:,3,:]
nrrs443_tmp = dataAll[:,4,:]
nrrs490_tmp = dataAll[:,5,:]
nrrs555_tmp = dataAll[:,6,:]

plt.hist(pft)
plt.figure()
plt.subplot(2,2,1)
plt.hist( pft);
plt.xlabel("Classes");
plt.title("PFT ", fontsize = 11)

datasd = pd.DataFrame(np.delete(dataAllwithnana.T,2, axis = 1))
datasd = np.delete(dataAllwithnana.T,2,axis=1)
datasd = datasd.T
dataR = datasd[0,:]
summ= np.sum(dataR,axis = 0)
ind = np.where(~np.isnan(summ))
dataR = dataR[:,ind]
dataR  = np.reshape(dataR, (6, len(ind[0]))) #np.reshape(ind)[1]))
cols = ["CHL-OC5", "SST","NRRS412","NRRS443","NRRS490", "NRRS555"]

datasd1 = pd.DataFrame(dataR.T, columns = cols)

cor = datasd1.corr()
#plt.matshow(cor)
#plt.
plt.figure()
corr = np.corrcoef(np.transpose(datasd1))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
        ax = sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), mask=mask, square=True, annot=True,
                         xticklabels=["CHL-OC5", "SST","NRRS412","NRRS443","NRRS490", "NRRS555"],
                         yticklabels=["CHL-OC5", "SST","NRRS412","NRRS443","NRRS490", "NRRS555"])
plt.title("Matrice de corrélations")
plt.show()


dataTrans = StandardScaler().fit_transform(datasd1)
dataTrans = pd.DataFrame(dataTrans, columns=cols)

cols1 = ["CHL-OC5", "SST","NRRS412","NRRS443","NRRS490", "NRRS555"]
pca = PCA(svd_solver='full')
acpdata = pca.fit_transform(dataTrans.T)
#fig, ax = plt.subplots()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Move left y-axis and bottim x-axis to centre, passing through (0,0)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
#ax.set_xlim(-400,400)
#ax.set_ylim(-400,400)
# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.scatter(acpdata[:,0], acpdata[:,1])
for i in range(6):
    plt.annotate(cols1[i],(acpdata[i,0], acpdata[i,1]))

print(pca.n_components_)
print(pca.explained_variance_)