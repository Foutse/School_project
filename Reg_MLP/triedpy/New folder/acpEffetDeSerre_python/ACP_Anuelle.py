# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 05:28:33 2018

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

def moyan(X,lesAnnees):
    """ MOYAN   :       Calcul des moyennes annuelles
        
        Xmoyan = moyan(X);
        
        En entree :
        -----------
        X     : Un tableau de données.
        annee : vecteur des annne 
        
        Le calcul des moyennes anuelles s'effecte pour chacune
        des colonnes de X (sur le critère du vecteur annee). """
    lesDifferentesAnnees = np.unique(lesAnnees)
    nbTotalDAnnees = len(lesDifferentesAnnees)
    nbTotalDeVariables = X.shape[1]
    resultat = np.zeros((nbTotalDAnnees,nbTotalDeVariables))
    for i,annee in enumerate(np.unique(lesDifferentesAnnees)):
        listeDIndices, = np.where(lesAnnees==annee)
        resultat[i,:] = X[listeDIndices,:].mean(axis=0)
    return resultat, lesDifferentesAnnees


def centred(X,moy=None,ecart_type=None,coef=None):
    """CENTRED :
       fonction de normalisation par centrage et reduction 
       avec la formule XN = coef * (X-moy) / std
       ou X sont les donnees, Moy leurs moyennes, 
       Std leurs ecarts types et Coef un coefficient."""

    nb_ind, nb_var = X.shape
    if moy == None:
        moy = X.mean(axis=0)
    else:
        assert(moy.isvector() and len(moy)==nb_var)
    if ecart_type == None:
        ecart_type = X.std(axis=0)
    else:
        assert(ecart_type.isvector() and len(ecart_type)==nb_var)
    if coef == None:
        coef = np.ones((1,nb_var))
    else:
        assert(coef.isvector() and len(coef)==nb_var)

    XN  = coef*(X - moy)/ecart_type;

    return XN                   
                    
def acp(X):
    """ ACP     :       ACP (Analyse en Composantes Principales)
        VEPU, CP, VAP = acp(X);
        
        En entree :
        -----------
        X : Matrice des données (individus en ligne, variables
                                 en colonnes)
        En Sortie :
        -----------
        VEPU : Axes principaux de M-norme 1. Ce sont les vecteurs
            propres de la matrice de covariance des données centrées.
        CP   : Composantes principales : coordonnées sur les nouveaux axes
            de l'ACP.
        VAP  : Valeurs propres des la matrice de covariance des données
            centrées."""
    # on centre les donnees (au cas ou)
    Xc = X - X.mean(axis=0)
    # determination valeurs et vecteurs  propres
    # de la matrice d'inertie
    VAP, VEPU = lna.eig(Xc.T@Xc);
    # on reordonne les vap et vep (au cas ou)
    idx  = VAP.argsort()[::-1]
    VAP  = VAP[idx];
    VEPU = VEPU[:,idx]; # on permute les colonnes
        
    # on a donc les directions et les inerties des nouveaux axes
        
    # determination des nouvelles coordonnées
    CP = Xc@VEPU
    # python travaille avec des tableaux de tableaux 

        
    return VEPU, CP, VAP

def phinertie(ValPropres):
    """PHINERTIE :       Calcul et cumuls des pourcentages 
       d'inertie de valeurs (propres) positives et présentation
       graphique par histogramme.
       
       PcentIg = phinertie(ValPropres);
       
       Entrées :
       ---------
       ValPropres : Un vecteur de valeurs (propres)
       
       Sorties :
       ---------
       PcentIg : Tableau à 2 colonnes dont les lignes sont dans le meme
                   ordre que celles des valeurs :           :
                   - 1ère colonne : pourcentage d'inertie
                   - 2ème colonne : pourcentage d'inertie cumulée"""
    p = np.size(ValPropres);
    sumI      = np.sum(ValPropres);
    INERTIE   = ValPropres/sumI;  
    ICUM      = np.cumsum(INERTIE); 
    #print('\nInertie=', INERTIE, '\nInertie cum.=',ICUM);
    plt.figure();
    index = np.arange(p);
    plt.bar(index+1,INERTIE);
    plt.plot(index+1, ICUM, 'r-*');
    for i in range(p) :
        plt.text(i+0.75,INERTIE[i],"%.4f" % (INERTIE[i]));
        plt.text(i+0.75,ICUM[i],"%.4f" % (ICUM[i]));
    plt.legend(["Inertie cumulée", "Inertie"], loc="upper left");
    plt.xlabel("Axes principaux");
    plt.ylabel("Poucentage d'Inertie des valeurs propres");
    return INERTIE, ICUM;

def qltctr2 (XU, VAPU) :
    ''' QLT, CTR = qltctr2 (XU, VAPU);
    | Dans le cadre d'une acp, dont XU sont les nouvelles coordonnées des
    | individus, et VAPU les valeurs propres, qltctr2 calcule et retourne : 
    | - QLT : Les qualités de réprésentation des individus par les axes
    | - CTR : Les contributions des individus à la formation des axes
    '''
    # Qualité de représentation et Contribution des Individus
    p       = np.size(VAPU); 
    C2      = XU*XU; 
    CTR     = C2 / VAPU; 
    dist    = np.sum(C2,1);          
    Repdist = nm.repmat(dist,p,1);
    QLT     = C2 / Repdist.T; 
    return QLT, CTR;

def cerclecor () :
    ''' Trace un cercle (de rayon 1 et de centre 0) pour le cercle des corrélations
    '''
    plt.figure();
    # Construire et tracer un cercle de rayon 1 et de centre 0
    t = np.linspace(-np.pi, np.pi, 50); 
    x = np.cos(t);
    y = np.sin(t);
    plt.plot(x,y,'-r');  # trace le cercle 
    plt.axis("equal");
    # Tracer les axes
    xlim = plt.xlim(); plt.xlim(xlim); 
    plt.plot(xlim, np.zeros(2),'k');
    ylim = plt.ylim(); plt.ylim(ylim);  
    plt.plot(np.zeros(2),ylim,'k');


def corcer(X,CP,pa,pb,varnames,shape='o',coul='b',markersize=8, fontsize=11):
    """ CORCER :         Cercle de corrélations : Représentation sur un cercle
        des coefficients de corrélation entre deux nouvelles variables et
        l'ensemble des variables initiales (colonnes de la matrice X).
        Ces coefficients sont représentés par un vecteur flèche à partir de 
        l'origine 0.
        
        corcer(X,CP,pa,pb,varnames,Col,disp);
        
        Entrée :
        -------- 
        X        : Matrice des données d'origines (variables en colonne)
        CP       : Composantes principales résultants d'une ACP (composante en colonnes).
                   Doit etre de meme dimension que X.
        pa, pb   : indice (n°) des 2 axes à traiter (i.e. des 2 nouvelles variables)
                   varnames : Tableau des labels des variables d'origines à associer aux variables
                   de X. Il doit donc avoir la meme longueur que X a de colonnes.
                   Des variables peuvent avoir le meme nom de label auquel cas leurs
                   vecteurs flèches seront tracés dans la meme couleur (c.f. paramètre
                   Col)
        Col      : Map de couleur pour les traits des vecteurs flèches à associer dans 
                   l'ordre à chaque "cas" de nom de variables (varnames). Ansi, si par 
                   exemple, plusieurs variables sont associés au meme nom (par
                   exemple 'AVRIL'), toutes les flèches des coefficients de ces variables 
                   auront la meme couleur. La taille de la map doit au moins etre égale
                   au nombre de "cas" de noms de variable différents.
                   Par défaut : les flèches seront de couleur bleu.
        disp     : Choix d'un type d'affichage :                  
                   1 : tous les vecteurs avec un seul label par "cas de nom de variable"  
                       itionné sur le vecteur moyen du "cas"; colorbar  affichée    
                   2 : un subplot par "cas"; pas de label, pas de colorbar
                   3 : vecteurs moyens des "cas" ; label sur les vecteurs moyens; colorbar affichée 
                   Par defaut : comme 1 sans la colorbar. """
                   
    p = np.size(CP,1);
    cerclecor();
    # Déterminer les corrélations et les ploter
    XUab = CP[:,[pa,pb]];
    W = np.concatenate((X,XUab), axis=1);
    R =  np.corrcoef(W.T);
    a = R[0:p,p]; 
    b = R[0:p,p+1];
    #
    #plt.plot(a,b,shape,color=coul,markersize=markersize);    
    for k in range(len(varnames)):
            plt.arrow(0, 0, a[k], b[k],length_includes_head=True, head_width=0.05, head_length=0.1, fc='k', ec='k')
            # Ornementation
            plt.text(a[k], b[k], varnames[k])#,fontsize=fontsize);
        
    #for i in range(p) :
    #    plt.text(a[i],b[i], varnames[i],fontsize=fontsize);
    #
    plt.xlabel("axe {:d}".format(pa+1),fontsize=fontsize);
    plt.ylabel("axe {:d}".format(pb+1),fontsize=fontsize);
    plt.title("ACP : Cercle des corrélations plan {:d}-{:d}".format(pa+1, pb+1), fontsize=fontsize);

    return None

def acpnuage(CP,cpa,cpb,Xcol,indnames,K,k0=.5,names=False,xoomK=100,shape='>',markersize=5, fontsize=8):
    """ ACPNUAGE :          Nuage des individus d'une ACP sur le plan de
        2 composantes d'indices cpa et cpb. Les points du nuage peuvent
        etre associés à une échelle de couleurs selon un vecteur donné. 
        Les marqueurs des points peuvent etre 2 triangles proportionnés 
        par les composantes d'une matrice à 2 colonnes (K) et d'un 
        facteur de taille (xoomK). Ces triangles sont orientés selon 
        l'abscisse et l'ordonnée de la façon suivante vers la droite 
        pour l'abscisse et vers le haut pour l'ordonnée
        
        acpnuage(CP,cpa,cpb,Xcol,indnames,K,xoomK);

        Entrées :
        ---------
        CP       : Matrice des composantes principales de l'ACP.
        cpa, cpb : Indices des 2 composantes principales (en colonne) à prendre
                   en compte pour former le plan du nuage des individus.
        Xcol     : Vecteur sur lequel sera établie l'échelle de couleur
                   des points. Si Xcol est vide (==[]), les points seront
                   tous de la meme couleur (bleu).
        indnames : Tableaux des noms des individus (points) avec lesquels
                   ils pourront etre labélisés si le tableau n'est pas vide ([]).
                   Dans ce cas, il doit avoir la meme longueur que CP.
        K        : Matrice de la meme taille que CP (par exemple une matrice
                   des contributions, ou des qualités de représentation) dont 
                   les colonnes d'indices cpa et cpb seront utilisées pour proportionner
                   les marqueurs triangles selon l'abscisse et l'ordonnées.
                   K doit avoir les memes dimensions que CP
        xoomK    : facteur de grossissement pour les marqueurs en formes de triangles
                   à déterminer expérimentalement de façon convenable.
        
        Remarque : Pour avoir un point de comparaison, lorsque les marqueurs triangles 
        ---------- sont utilisés, nous avons représenté, en bas et à gauche de la figure, 
                   3 triangles noirs dont les tailles correspondent respectivement à des
                   facteurs de 1.0 , 0.5 et 0.1 du facteur xoomK indiqué. """
    
    #plt.figure();
    #plt.plot(CP[:,cpa], CP[:,cpb],shape,color=Xcol,markersize=K);

    # mise en forme associées a K et Xcol
    # --------------------------
    
    # couleur en fonction de Xcol
    listeDesCouleurs = cm.jet((Xcol-Xcol.min())/(Xcol.max()-Xcol.min()))
    
    # taille des elements relativement aux valeurs de K 
    area = K[:,(cpa,cpb)].sum(axis=1)
    # limitation de l'affichage aux valeurs significatives
    mask = area < k0
    area = np.ma.masked_where(mask, area)*xoomK
    
    if True: # if False: # 
        # rotation de la marque raltive aux axes
        angle = np.arccos(K[:,cpa]/np.sqrt((K[:,(cpa,cpb)]**2).sum(axis=1)))/np.pi*180
        #plt.hist(angle)
        t = [mpl.markers.MarkerStyle(marker='>') for i in range(len(angle))]
        for i in range(len(t)):
            t[i].transform = t[i].get_transform().rotate_deg(45)
            
        # affichage
        listeDIndices, = np.where(np.logical_not(mask))
        for i in listeDIndices : # for i in range(CP.shape[0]) :
            t = mpl.markers.MarkerStyle(marker=shape)
            t._transform = t.get_transform().rotate_deg(angle[i])
            plt.scatter(CP[i,cpa], CP[i,cpb],marker=t,c='none',s=area[i], edgecolors=listeDesCouleurs[i],cmap=cm.jet)
            if names == True :
                plt.text(CP[i,cpa], CP[i,cpb], indnames[i],fontsize=fontsize);
    else:
        # affichage
        listeDIndices, = np.where(np.logical_not(mask))
        for i in listeDIndices : # for i in range(CP.shape[0]) :
            plt.scatter(CP[:,cpa], CP[:,cpb],c='none',s=area,marker=shape, edgecolors=listeDesCouleurs,cmap=cm.jet)
        if names == True :
            listeDIndices, = np.where(np.logical_not(mask))
            for i in listeDIndices : # for i in range(CP.shape[0]) :
                plt.text(CP[i,cpa], CP[i,cpb], indnames[i],fontsize=fontsize);
    
#    for m in ['d', '+', '|']:
#        for i in range(5):
#            a1, a2  = np.random.random(2)
#            angle = np.random.choice([180, 45, 90, 35])
#    
#            # make a markerstyle class instance and modify its transform prop
#            t = mpl.markers.MarkerStyle(marker=m)
#            t._transform = t.get_transform().rotate_deg(angle)
#            plt.scatter((a1), (a2), marker=t, s=100)

    # Tracer les axes
    xlim = plt.xlim(); plt.xlim(xlim);
    plt.plot(xlim, np.zeros(2),'k');
    ylim = plt.ylim(); plt.ylim(ylim); 
    plt.plot(np.zeros(2),ylim,'k');
    # Ornementation
    plt.xlabel("axe {:d}".format(cpa+1),fontsize=fontsize);
    plt.ylabel("axe {:d}".format(cpb+1),fontsize=fontsize);
    
    N = len(np.unique(Xcol))
    vmin, vmax = Xcol.min(), Xcol.max()
    cmap = plt.get_cmap('jet',N)
    norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ticks=np.linspace(vmin,vmax,N)[0::3],boundaries=np.arange(vmin,vmax+1,1))


    return None
def chargerLesDonnees_notes():
    notes = pd.read_table("notes.csv",sep='\t',index_col=0)
    nomDesIndividus = notes.index
    nomDesVariables = notes.columns
    Z = notes.values
    return Z, nomDesIndividus, noms_des_variables

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
    temps = np.arange(1982,2010) #np.arange(1,30) #annee #+ (mois-1)/12
    #nombre_de_pas_de_temps = len(temps)  # Nombre de pas de temps
    
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
   # noms_des_mois        = ('janv','fév','mars','avril','mai','juin','juil','aout','sept','oct','nov','déc')
   # nombre_de_mois      = len(noms_des_mois)
    noms_des_annee    = np.arange(1982,2010)
    nombre_de_annee    = len(noms_des_mois)#range(1,29)
    # Calcul et affichage des climatologies (par ville )
    fig = plt.figure(figsize=(18, 6))
    #Z = np.zeros((nombre_de_pas_de_temps,nombre_de_variables))
   # climatologie = np.zeros((nombre_de_annee,nombre_de_variables))
    # Boucle sur les villes
    for k in range(0,nombre_de_lieux):
        # aggregation des donnees pour la keme ville
        Z = np.column_stack((clim_t2[:,k], clim_tcc[:,k], clim_lsp[:,k], 
            clim_cp[:,k], clim_ssr[:,k], clim_co2[:,0]))
        
        # Centrage et Réduction des variables
        X = centred(Z)
        #X = (Z-Z.mean(axis=0))/Z.std(axis=0);
        
       # Calcul de la climatologie (moyenne par mois)
       # for m in range(nombre_de_annee):
        climatologie1, annee = moyan(X, annee)#  np.mean(X[annee == m,:],axis=0)
        #t2moyan=[]
#        for i in range(29):
#            climatologie[i,:] =np.mean(climatologie[k[i]:k[i+1]])   
#        
        # Affichage
        plt.subplot(3,3,k+1)
        plt.plot(climatologie1,'-o')
        plt.text(.5, .00, noms_des_lieux[k], horizontalalignment='center', 
             verticalalignment='bottom', transform=plt.gca().transAxes)
        plt.xticks(np.arange(nombre_de_annee),noms_des_annee,fontsize=8)
        plt.grid(True)
    #fig.subplots_adjust(bottom=-2, wspace=0.2)
    plt.legend(noms_des_variables,loc='upper center', bbox_to_anchor=(-0.5, -0.2),fancybox=False, shadow=False, ncol=nombre_de_variables);
    plt.suptitle("Climatologie annuelle (annee moyen) des villes (données centrées et réduites)"); 


def moyannee (Z,annee,noms_des_variables,nomDeLaVille):
    climatologie1, annee = moyan(Z, annee)
    X =  centred(climatologie1)
    X = X.T
    nombre_des_variables = len(noms_des_variables)
    DiffAnn = np.unique(annee)
    nombre_de_annee = len(DiffAnn)
    
    plt.figure(figsize = (18,8))
    for i in range(nombre_des_variables):
        plt.plot(X[i], '-o')
    plt.xticks(np.arange(nombre_de_annee),range(0,nombre_de_annee),fontsize=8)
    plt.grid(True)
    plt.legend(noms_des_variables,loc='best',fancybox=False, shadow=False, ncol=nombre_des_variables);
    plt.suptitle("Climatologie annuelle (annee moyen) des villes (données centrées et réduites)" + nomDeLaVille); 
    plt.xlabel("Annee")
    plt.show()
#t2moyan=[]
#for i in range(29):
#    t2moyan.append(np.mean(climatologie[k[i]:k[i+1]]))
#t2moyan

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   xclimmens()
 
   nomDeLaVille = 'Reykjavik'
   Z, mois, annee, noms_des_variables = chargerLesDonnees_effetDeSerre(nomDeLaVille)
   #Z, noms_des_individus, noms_des_variables = chargerLesDonnees_notes()
   xoomK1 = 100
   xoomK2 = 5000
   
   # realisation de moyennes annuelles
   if True: #if False:  if True:
       Z, annee = moyan(Z,annee)
      # del mois # n'a plus de sens dans le cadre de moyennes annuelles
       xoomK1 = 1000 
       xoomK2 = 5000
   
   moyannee (Z,annee,noms_des_variables,nomDeLaVille)
    
   nombre_individus = Z.shape[0]
   X = centred(Z)
   VEPU, CP, VAP = acp(X)
   INERTIE, ICUM = phinertie(VAP)
   corcer(X,CP,0,1,noms_des_variables,shape='o',coul='b',markersize=8, fontsize=11)
   QLT, CTR = qltctr2 (CP, VAP)
   #print('qualite: ',QLT)
   #print('contribution: ',CTR)
  
    
#   if 'mois' in locals():
#       indMois = []
#       for i in mois:
#           indMois.append("{:.0f}".format(i))
   if 'annee' in locals():
       annee = annee - annee.min()
       indAnnee = []
       annee = annee - annee.min()
       indAnnee = []
       for i in annee:
           indAnnee.append("{:.0f}".format(i))
      
#   if 'mois' in locals() or 'annee' in locals():
#       fig = plt.figure(figsize=(18, 6))
#       plt.suptitle("Affichage en fonction de la qualité"); 
#   if 'mois' in locals() and 'annee' in locals():
#       plt.subplot(2,1,1)
#       acpnuage(CP,0,1,Xcol=mois,indnames=indMois,K=QLT,k0=.5,names=True,xoomK=xoomK1)
#       #plt.axis('equal')
#       plt.subplot(2,1,2)
#       acpnuage(CP,0,1,Xcol=annee,indnames=indAnnee,K=QLT,k0=.5,names=True,xoomK=xoomK1)
#       #plt.axis('equal')
#   elif 'mois' in locals():
#       acpnuage(CP,0,1,Xcol=mois,indnames=indMois,K=QLT,k0=.5,names=True,xoomK=xoomK1)
       #plt.axis('equal')
   elif 'annee' in locals():
       acpnuage(CP,0,1,Xcol=annee,indnames=indAnnee,K=QLT,k0=.5,names=True,xoomK=xoomK1)
       #plt.axis('equal')
          
#   if 'mois' in locals() or 'annee' in locals():
#       fig = plt.figure(figsize=(18, 6))
#       plt.suptitle("Affichage en fonction de la contribution"); 
#   if 'mois' in locals() and 'annee' in locals():
#       plt.subplot(2,1,1)
#       acpnuage(CP,0,1,Xcol=mois,indnames=indMois,K=CTR,k0=1/nombre_individus,names=True,xoomK=xoomK2)
#       #plt.axis('equal')
#       plt.subplot(2,1,2)
#       acpnuage(CP,0,1,Xcol=annee,indnames=indAnnee,K=CTR,k0=1/nombre_individus,names=True,xoomK=xoomK2)
#       #plt.axis('equal')
#   elif 'mois' in locals():
#       acpnuage(CP,0,1,Xcol=mois,indnames=indMois,K=CTR,k0=1/nombre_individus,names=True,xoomK=xoomK2)
       #plt.axis('equal')
   elif 'annee' in locals():
       acpnuage(CP,0,1,Xcol=annee,indnames=indAnnee,K=CTR,k0=1/nombre_individus,names=True,xoomK=xoomK2)
       #plt.axis('equal')
     