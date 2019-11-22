# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 00:29:39 2018

@author: FOUTSE
"""

import math
import numpy as np
import numpy.linalg as lna
import scipy.io
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.matlib as nm
from matplotlib import cm        # module palettes de couleurs
import pandas as pd
from sklearn.decomposition import PCA
np.set_printoptions(precision=1)
#- xclimmens : Charge les données puis produit des tracés par ville des climatologies mensuelles
#pour les variables t2, tcc, lsp, cp, ssr et co2.

def chargerLesDonnees_effetDeSerre(nomDeLaVille):
    # Chargement des données  (période 1/1982 à 10/2011)
    clim_t2  = np.array(scipy.io.loadmat('clim_t2C_J1982D2010')['clim_t2'])  # Temperature at 2 meters (degC) 
    clim_tcc = np.array(scipy.io.loadmat('clim_tcc_J1982D2010')['clim_tcc']) # Total cloud cover (0-1)
    clim_lsp = np.array(scipy.io.loadmat('clim_lsp_J1982D2010')['clim_lsp']) # Large scale precipitation (m) 
    clim_cp  = np.array(scipy.io.loadmat('clim_cp_J1982D2010')['clim_cp'])   # Convective precipitation (m) 
    clim_ssr = np.array(scipy.io.loadmat('clim_ssr_J1982D2010')['clim_ssr']) # Surface solar radiation ((W/m^2)s)
    clim_co2 = np.array(scipy.io.loadmat('clim_co2_J1982D2010')['clim_co2']) # CO2 ppm
    
    # on met mois et annees de cote
    assert(np.all(clim_t2[:,0] == clim_tcc[:,0]) and np.all(clim_t2[:,0] == clim_lsp[:,0])
           and np.all(clim_t2[:,0] == clim_cp[:,0]) and np.all(clim_t2[:,0] == clim_ssr[:,0])
           and np.all(clim_t2[:,0] == clim_co2[:,0]))
    annee = clim_t2[:,0]
    
    assert(np.all(clim_t2[:,1] == clim_tcc[:,1]) and np.all(clim_t2[:,1] == clim_lsp[:,1])
           and np.all(clim_t2[:,1] == clim_cp[:,1]) and np.all(clim_t2[:,1] == clim_ssr[:,1])
           and np.all(clim_t2[:,1] == clim_co2[:,1]))
    mois = clim_t2[:,1]
    temps = annee + (mois-1)/12
    nombre_de_pas_de_temps = len(temps)  # Nombre de pas de temps
    
    clim_t2  = np.delete(clim_t2, [0,1], axis=1)
    clim_tcc = np.delete(clim_tcc, [0,1], axis=1)
    clim_lsp = np.delete(clim_lsp, [0,1], axis=1)
    clim_cp  = np.delete(clim_cp, [0,1], axis=1)
    clim_ssr = np.delete(clim_ssr, [0,1], axis=1)
    clim_co2 = np.delete(clim_co2, [0,1], axis=1)
    
    # informations associees aux donnees
    noms_des_variables   = ('t2','tcc','lsp','cp','ssr','CO2') 
    nombre_de_variables = len(noms_des_variables)   # Nombre de variables
    noms_des_lieux       = ('Reykjavik','Oslo','Paris','New York','Tunis','Alger','Beyrouth','Atlan','Dakar')
    
    k = noms_des_lieux.index(nomDeLaVille)
    Z = np.column_stack((clim_t2[:,k], clim_tcc[:,k], clim_lsp[:,k], 
        clim_cp[:,k], clim_ssr[:,k], clim_co2[:,0]))
    
    return Z, mois, annee, noms_des_variables
    
def xclimmens():
    """XCLIMMENS :
       Climatologie mensuelle par ville des variables :
       t2, tcc, lsp, cp, ssr et CO2."""

    # Chargement des données  (période 1/1982 à 10/2011)
    clim_t2  = np.array(scipy.io.loadmat('clim_t2C_J1982D2010')['clim_t2'])  # Temperature at 2 meters (degC) 
    clim_tcc = np.array(scipy.io.loadmat('clim_tcc_J1982D2010')['clim_tcc']) # Total cloud cover (0-1)
    clim_lsp = np.array(scipy.io.loadmat('clim_lsp_J1982D2010')['clim_lsp']) # Large scale precipitation (m) 
    clim_cp  = np.array(scipy.io.loadmat('clim_cp_J1982D2010')['clim_cp'])   # Convective precipitation (m) 
    clim_ssr = np.array(scipy.io.loadmat('clim_ssr_J1982D2010')['clim_ssr']) # Surface solar radiation ((W/m^2)s)
    clim_co2 = np.array(scipy.io.loadmat('clim_co2_J1982D2010')['clim_co2']) # CO2 ppm
    
    # on met mois et annees de cote
    assert(np.all(clim_t2[:,0] == clim_tcc[:,0]) and np.all(clim_t2[:,0] == clim_lsp[:,0])
           and np.all(clim_t2[:,0] == clim_cp[:,0]) and np.all(clim_t2[:,0] == clim_ssr[:,0])
           and np.all(clim_t2[:,0] == clim_co2[:,0]))
    annee = clim_t2[:,0]
    
    assert(np.all(clim_t2[:,1] == clim_tcc[:,1]) and np.all(clim_t2[:,1] == clim_lsp[:,1])
           and np.all(clim_t2[:,1] == clim_cp[:,1]) and np.all(clim_t2[:,1] == clim_ssr[:,1])
           and np.all(clim_t2[:,1] == clim_co2[:,1]))
    mois = clim_t2[:,1]
    temps = annee + (mois-1)/12
    nombre_de_pas_de_temps = len(temps)  # Nombre de pas de temps
    
    clim_t2  = np.delete(clim_t2, [0,1], axis=1)
    clim_tcc = np.delete(clim_tcc, [0,1], axis=1)
    clim_lsp = np.delete(clim_lsp, [0,1], axis=1)
    clim_cp  = np.delete(clim_cp, [0,1], axis=1)
    clim_ssr = np.delete(clim_ssr, [0,1], axis=1)
    clim_co2 = np.delete(clim_co2, [0,1], axis=1)
    
    # informations associees aux donnees
    noms_des_variables   = ('t2','tcc','lsp','cp','ssr','CO2') 
    nombre_de_variables = len(noms_des_variables)   # Nombre de variables
    noms_des_lieux       = ('Reykjavik','Oslo','Paris','New York','Tunis','Alger','Beyrouth','Atlan','Dakar')
    nombre_de_lieux     = len(noms_des_lieux)       # Nombre de villes
    noms_des_mois        = ('janv','fév','mars','avril','mai','juin','juil','aout','sept','oct','nov','déc')
    nombre_de_mois      = len(noms_des_mois)
    
    # Calcul et affichage des climatologies (par ville )
    fig = plt.figure(figsize=(18, 6))
    #Z = np.zeros((nombre_de_pas_de_temps,nombre_de_variables))
    climatologie = np.zeros((nombre_de_mois,nombre_de_variables))
    # Boucle sur les villes
    for k in range(0,nombre_de_lieux):
        # aggregation des donnees pour la keme ville
        Z = np.column_stack((clim_t2[:,k], clim_tcc[:,k], clim_lsp[:,k], 
            clim_cp[:,k], clim_ssr[:,k], clim_co2[:,0]))
        
        # Centrage et Réduction des variables
        X = centred(Z)
        #X = (Z-Z.mean(axis=0))/Z.std(axis=0);
        
        # Calcul de la climatologie (moyenne par mois)
        for m in range(nombre_de_mois):
            climatologie[m,:] = np.mean(X[mois==m+1,:],axis=0)
        
        # Affichage
        plt.subplot(3,3,k+1)
        plt.plot(climatologie,'-o')
        plt.text(.5, .00, noms_des_lieux[k], horizontalalignment='center', 
             verticalalignment='bottom', transform=plt.gca().transAxes)
        plt.xticks(np.arange(nombre_de_mois),noms_des_mois,fontsize=8)
        plt.grid(True)
    #fig.subplots_adjust(bottom=-2, wspace=0.2)
    plt.legend(noms_des_variables,loc='upper center', bbox_to_anchor=(-0.5, -0.2),fancybox=False, shadow=False, ncol=nombre_de_variables);
    plt.suptitle("Climatologie mensuelle (mois moyen) des villes (données centrées et réduites)"); 


xclimmens()


#- centred : Normalisation par centrage réduction.
nomDeLaVille = 'Reykjavik'
Z, mois, annee, noms_des_variables = chargerLesDonnees_effetDeSerre(nomDeLaVille)