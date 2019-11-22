import time
import sys
import numpy as np
import matplotlib.pyplot as plt

#======================== k-plus proche voisins =========================
def kppvo (Xref,Yref,Xtst,k,dist=0) :
    '''KLASS = kppvo(Xref,Yref,Xtst,k,dist)
    | Algorithme des k plus proches voisins 
    | En entrée :
    | Xref : Ensemble de référence
    | Yref : les indices de classe de référence (i.e. des éléments de l'ensemble
    |        de référence). Ces indices doivent commencer à 1 (>0) et non pas
    |        à partir de zéro.   : 
    | Xtst : Un ensemble (de même dimension que Xref) dont on veut classifier les
    |        éléments par l'algorithme des k plus proches voisins 
    | k    : Le nombre de voisins à considérer
    | dist : Distance à utiliser : 0 : Euclidienne (par défaut) ; sinon: Mahalanobis
    | En sortie :
    | KLASS : Classes des éléments de Xtst
    '''
    if min(Yref)<=0 :
        print("kppvo: Les indices de classe de référence (Yref) doivent être > 0");
        sys.exit(0);
        
    nX,  dim = np.shape(Xtst);
    nXi, d2  = np.shape(Xref);
    c        = max(Yref);

    if d2 != dim :
        print("kppvo: Xtst doit avoir la même dimension de Xest");
        sys.exit(0);
    if np.size(Xref,0) != np.size(Yref) :
        print("kppvo, Xref et Yest doivent avoir le même nombre d'élément");
        sys.exit(0);
      
    Yref=Yref-1; # parce que les indices commence à 0 ...?
    KLASS = np.zeros(nX); # Init Tableau des classes des éléments

    # Construction de la METRIQUE
    SIGMA = np.zeros((int(c),dim,dim)); # Init.  des matrices de COVARIANCE (par classe)
    if dist!=0 : # Distance de MAHALANOBIS
        for i in np.arange(int(c)) :      # Calcul des matrices de COVARIANCE (par classe)
            ICi         = np.where(Yref==i)[0]; # Indices des élts de la classe i
            XrefCi      = Xref[ICi,:];          # Ens de Ref pour la classe i
            sigma       = np.cov(XrefCi.T, ddof=1);
            sigmamoins  = np.linalg.inv(sigma);
            SIGMA[i,:,:]= sigmamoins;
    else :  # Distance euclidienne en prenant la matrice identité
        for i in np.arange(int(c)) :
            SIGMA[i,:,:]= np.eye(dim);
    # PS dans SIGMA les Metriques Mi se retrouvent empilées par classe.
        
    # DECISION
    for i in np.arange(nX) : # Pour chaque elt de l'ens de TEST
        D = np.zeros(nXi);

        for j in np.arange(nXi) : # Pour chaque elt de l'ens de référence
            cl   = Yref[j];
            M    = np.dot(Xtst[i,:]-Xref[j,:], SIGMA[int(cl),:,:]);
            D[j] = np.dot(M, Xtst[i,:]-Xref[j,:]);            

        # Tri des distances dans l'ordre du +petit au +grand
        I = sorted(range(len(D)), key=lambda k: D[k])
        C         = Yref[I];   # On ordonne les classes selon se tri
        classeppv = C[0:k];    # On garde les k premières classes qui correspondent 
                               # donc aux k plus proches voisins dans l'ens d'APP
                               
        # Vote majoritaire : 
        nc = np.zeros(int(c));      # Init (raz) du tableau de comptage du nombre de classes
                               # pour le vote majoritaire.
                               
        for j in np.arange(k):     # comptage des classes
            nc[int(classeppv[j])] +=1;  # (incrémentation)

        Imax = np.argmax(nc);      # On regarde la classe qui a le plus de vote
        KLASS[i] = Imax;           # que l'on affecte à TESTi.

    KLASS = KLASS+1; # Pour revenir à l'indicage initial.
    return KLASS
#
#==================================================================


#kppv aleatoirement
    

def kppvo_a (Xref,Yref,Xtst,k,dist=0) :
    '''KLASS = kppvo(Xref,Yref,Xtst,k,dist)
    | Algorithme des k plus proches voisins 
    | En entrée :
    | Xref : Ensemble de référence
    | Yref : les indices de classe de référence (i.e. des éléments de l'ensemble
    |        de référence). Ces indices doivent commencer à 1 (>0) et non pas
    |        à partir de zéro.   : 
    | Xtst : Un ensemble (de même dimension que Xref) dont on veut classifier les
    |        éléments par l'algorithme des k plus proches voisins 
    | k    : Le nombre de voisins à considérer
    | dist : Distance à utiliser : 0 : Euclidienne (par défaut) ; sinon: Mahalanobis
    | En sortie :
    | KLASS : Classes des éléments de Xtst
    '''
    if min(Yref)<=0 :
        print("kppvo: Les indices de classe de référence (Yref) doivent être > 0");
        sys.exit(0);
        
    nX,  dim = np.shape(Xtst);
    nXi, d2  = np.shape(Xref);
    c        = max(Yref);

    if d2 != dim :
        print("kppvo: Xtst doit avoir la même dimension de Xest");
        sys.exit(0);
    if np.size(Xref,0) != np.size(Yref) :
        print("kppvo, Xref et Yest doivent avoir le même nombre d'élément");
        sys.exit(0);
      
    Yref=Yref-1; # parce que les indices commence à 0 ...?
    KLASS = np.zeros(nX); # Init Tableau des classes des éléments

    # Construction de la METRIQUE
    SIGMA = np.zeros((int(c),dim,dim)); # Init.  des matrices de COVARIANCE (par classe)
    if dist!=0 : # Distance de MAHALANOBIS
        for i in np.arange(int(c)) :      # Calcul des matrices de COVARIANCE (par classe)
            ICi         = np.where(Yref==i)[0]; # Indices des élts de la classe i
            XrefCi      = Xref[ICi,:];          # Ens de Ref pour la classe i
            sigma       = np.cov(XrefCi.T, ddof=1);
            sigmamoins  = np.linalg.inv(sigma);
            SIGMA[i,:,:]= sigmamoins;
    else :  # Distance euclidienne en prenant la matrice identité
        for i in np.arange(int(c)) :
            SIGMA[i,:,:]= np.eye(dim);
    # PS dans SIGMA les Metriques Mi se retrouvent empilées par classe.
        
    # DECISION
    for i in np.arange(nX) : # Pour chaque elt de l'ens de TEST
        D = np.zeros(nXi);

        for j in np.arange(nXi) : # Pour chaque elt de l'ens de référence
            cl   = Yref[j];
            M    = np.dot(Xtst[i,:]-Xref[j,:], SIGMA[int(cl),:,:]);
            D[j] = np.dot(M, Xtst[i,:]-Xref[j,:]);            

        # Tri des distances dans l'ordre du +petit au +grand
        I = sorted(range(len(D)), key=lambda k: D[k])
        C         = Yref[I];   # On ordonne les classes selon se tri
        classeppv = C[0:k];    # On garde les k premières classes qui correspondent 
                               # donc aux k plus proches voisins dans l'ens d'APP
                               
        # Vote majoritaire : 
        nc = np.zeros(int(c));      # Init (raz) du tableau de comptage du nombre de classes
                               # pour le vote majoritaire.
                               
        for j in np.arange(k):     # comptage des classes
            nc[int(classeppv[j])] +=1;  # (incrémentation)
        
        Imax = np.argmax(nc);      # On regarde la classe qui a le plus de vote
        Imin = np.argmin(nc);
        if Imax == Imin  :          # que l'on affecte à TESTi
              KLASS[i] = np.random.choice(2) 
        else:             
             KLASS[i] = Imax;

    KLASS = KLASS+1; # Pour revenir à l'indicage initial.
    return KLASS
#