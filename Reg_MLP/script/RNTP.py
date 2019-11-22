# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 00:02:43 2018
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.cm

# If triedpy modules are not directly accessible by python,
# set the path where triedpy is located :
if 1 :
    TRIEDPY = "../";   # Mettre le chemin d'accès ... 
    sys.path.append(TRIEDPY);   # ... aux modules triedpy
from   triedpy import triedtools as tls
from   triedpy import trieddeep  as tdp
import TPB01_methodes
 
np.random.seed(0);    # (Re)positionnement du random (ou pas)

N = 1000
Xtest, Ytest = TPB01_methodes.schioler(500)
Xi, Yi = TPB01_methodes.schioler(N)
YC = TPB01_methodes.schiolerClean(Xi); # Valeur de la vraie fonction schioler (i.e. non bruitee) 
Xtrain = []
Ytrain = []
Xval = []
Yval = []
n=2
Xtrain = Xi[0:-1:n]
Ytrain = Yi[0:-1:n]
Xval = Xi[1::n]
Yval = Yi[1::n]
       
#1)
plt.figure();
plt.plot(Xtrain,Ytrain,'b*',markersize=8);    # training datas
plt.plot(Xval,Yval,'g*',markersize=8);    # validation datas
plt.title("Données d'apprentissage et données de validation", fontsize=15)
plt.legend(["Données d'apprentissage","Données de validation"],numpoints=1, fontsize=15);

plt.figure();
plt.plot(Xtest,Ytest,'k*', linewidth=1.5);   # test data
plt.title("Données de test", fontsize=15)

# > Initialisation du PMC
# Architecture : nombres de neurones par couches cachées
#2)

plt.figure();
#ax=plt.gca()
plt.plot(Xtrain,Ytrain,'b*',label='Train', markersize=8);    # training datas
plt.plot(Xi,YC,'k',label='True', linewidth=1.5);    # validation datas
plt.title("Prédictions après apprentissage avec différent nombre de neurones ", fontsize = 15)
for j in range(1,10, 2):
    m  = [j]; # Une couche cachée de 5 neurones
#
    # Fonctions de Transfert : mettre autant de Fi que de m+1
    F1 = "tah"; F2 = "lin"; 
    FF = [F1,F2];
    # Les poids
    WW   = tdp.pmcinit(Xtrain, Ytrain, m);
    # > Learning
    nbitemax = 10000; #1000
    dfreq    = 10; #200 
    dcurves  = 0;
    WW, it_minval = tdp.pmctrain(Xtrain,Ytrain,WW,FF, \
     nbitemax=nbitemax, dfreq=dfreq, dcurves=dcurves,  Xv=Xval,Yv=Yval);
    Y = tdp.pmcout(Xtrain,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    # Error et RMS en Apprentissage :
    errq, rms, = tls.errors(Ytrain, Y,["errq","rms"])
    print("\nApprentissage : (Erreur, RMS) = (%f, %f)" %(errq,rms));
    titre='pmc avec' + ' ' + str(j)
    plt.plot(Xtrain,Y, linewidth=1.5,label=titre);   
plt.legend(numpoints=1);

#3)
Nneurone = [1,3,5,7,9]
errRMS1 = []
errRMS2 = []
errRMS3 = []
for k in range(1,10,2):
    m= [k]
    F1 = "tah"; F2 = "lin"; 
    FF = [F1,F2];
#
    # Les poids
    WW   = tdp.pmcinit(Xtrain, Ytrain, m);
#
    # > Learning
#    nbitemax = 2000;
#    dfreq    = 200;  
    nbitemax = 10000; #1000
    dfreq    = 10; #200 
    dcurves  = 0;
    WW, it_minval = tdp.pmctrain(Xtrain,Ytrain,WW,FF, \
     nbitemax=nbitemax, dfreq=dfreq, dcurves=dcurves,  Xv=Xval,Yv=Yval);
    #plt.title("Minimisation sur la fonction de coût pour train et y",fontsize=16);
    Y1 = tdp.pmcout(Xtrain,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    errq, rms1, = tls.errors(Ytrain, Y1,["errq","rms"])
    errRMS1.append(rms1)
    Y2 = tdp.pmcout(Xval,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    errq, rms2, = tls.errors(Yval, Y2,["errq","rms"])
    errRMS2.append(rms2)
    Y3 = tdp.pmcout(Xtest,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    errq, rms3, = tls.errors(Ytest, Y3,["errq","rms"])
    errRMS3.append(rms3)
plt.figure();
plt.plot(Nneurone,errRMS1,'r*-',markersize=8);
plt.plot(Nneurone,errRMS2,'g*-',markersize=8);
plt.plot(Nneurone,errRMS3,'b*-',markersize=8);
plt.xlabel("Numbre de neurone", fontsize = 15) 
plt.legend(["RMS données d'apprentissage","RMS données validation","RMS données test"],numpoints=1, fontsize = 15);   
plt.title(" Erreurs (RMS) avec nos trois type de Données et les different nombre de neurones", fontsize = 15)
print(n, errRMS1)
print(n, errRMS2)
print(n, errRMS3)

#4)


N = 1000
Xtest, Ytest = TPB01_methodes.schioler(500)
Xi, Yi = TPB01_methodes.schioler(N)
YC = TPB01_methodes.schiolerClean(Xi); # Valeur de la vraie fonction schioler (i.e. non bruitee) 

n=3
Xtrain = Xi[0:-1:n]
Ytrain = Yi[0:-1:n]
Xval = Xi[1::n]
Yval = Yi[1::n]
errRMSt = [] 
errRMSv = []         
errRMS3 = []
plt.figure();
plt.subplot(1,2,1)   #ax=plt.gca()
plt.plot(Xtrain,Ytrain,'b*',label='Train', markersize=8);    # training datas
plt.plot(Xi,YC,'r',label='True', linewidth=1.5);    # validation datas
plt.title("Prédictions après apprentissage avec \n 1/3 de données pour différent nombre de neurones ", fontsize = 15)
#plt.legend(['train', 'true'], numpoints=1)
for j in range(1,10, 2):
    m  = [j]; # Une couche cachée de 5 neurones
#
    # Fonctions de Transfert : mettre autant de Fi que de m+1
    F1 = "tah"; F2 = "lin"; 
    FF = [F1,F2];
    # Les poids
    WW   = tdp.pmcinit(Xtrain, Ytrain, m);
    # > Learning
#    nbitemax = 2000;
#    dfreq    = 200;  
    nbitemax = 10000; #1000
    dfreq    = 10; #200 
    dcurves  = 0;
    WW, it_minval = tdp.pmctrain(Xtrain,Ytrain,WW,FF, \
     nbitemax=nbitemax, dfreq=dfreq, dcurves=dcurves,  Xv=Xval,Yv=Yval);
    Yt = tdp.pmcout(Xtrain,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    # Error et RMS en Apprentissage :
    errq, rmst, = tls.errors(Ytrain, Yt,["errq","rms"])
    errRMSt.append(rmst)
    Yv = tdp.pmcout(Xval,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    # Error et RMS en Apprentissage :
    errq, rmsv, = tls.errors(Yval, Yv,["errq","rms"])
    errRMSv.append(rmsv)
    Y = tdp.pmcout(Xtest,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    errq, rms, = tls.errors(Ytest, Y,["errq","rms"])
    errRMS3.append(rms)
    #print("\nApprentissage : (Erreur, RMS) = (%f, %f)" %(errq,rms));
    titre='pmc' + str(j)
    plt.plot(Xtrain,Yt, linewidth=1.5,label=titre);   
plt.legend(numpoints=1);
plt.subplot(1,2,2)   #ax=plt.gca()
#plt.figure();
plt.plot(Nneurone,errRMSt,'r*-',markersize=8);
plt.plot(Nneurone,errRMSv,'g*-',markersize=8);
plt.plot(Nneurone,errRMS3,'b*-',markersize=8); 
plt.legend(["RMS données d'apprentissage","RMS données validation","RMS données test"],numpoints=1);   
plt.title("RMS après apprentissage sur 1/3  \n de données  pour different nombre de neurones")
print(n, errRMSt)
print(n, errRMSv)
print(n, errRMS3)

n=4
Xtrain = Xi[0:-1:n]
Ytrain = Yi[0:-1:n]
Xval = Xi[1::n]
Yval = Yi[1::n]
errRMSt = [] 
errRMSv = []         
errRMS3 = []
plt.figure();
plt.subplot(1,2,1)   #ax=plt.gca()
plt.plot(Xtrain,Ytrain,'b*',label='Train', markersize=8);    # training datas
plt.plot(Xi,YC,'r',label='True', linewidth=1.5);    # validation datas
plt.title("Prédictions après apprentissage avec \n 1/4 de données pour différent nombre de neurones ", fontsize = 15)
#plt.legend(['train', 'true'], numpoints=1)
for j in range(1,10, 2):
    m  = [j]; # Une couche cachée de 5 neurones
#
    # Fonctions de Transfert : mettre autant de Fi que de m+1
    F1 = "tah"; F2 = "lin"; 
    FF = [F1,F2];
    # Les poids
    WW   = tdp.pmcinit(Xtrain, Ytrain, m);
    # > Learning
#    nbitemax = 2000;
#    dfreq    = 200; 
    nbitemax = 10000; #1000
    dfreq    = 10; #200 
    dcurves  = 0;
    WW, it_minval = tdp.pmctrain(Xtrain,Ytrain,WW,FF, \
     nbitemax=nbitemax, dfreq=dfreq, dcurves=dcurves,  Xv=Xval,Yv=Yval);
    Yt = tdp.pmcout(Xtrain,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    # Error et RMS en Apprentissage :
    errq, rmst, = tls.errors(Ytrain, Yt,["errq","rms"])
    errRMSt.append(rmst)
    Yv = tdp.pmcout(Xval,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    # Error et RMS en Apprentissage :
    errq, rmsv, = tls.errors(Yval, Yv,["errq","rms"])
    errRMSv.append(rmsv)
    Y = tdp.pmcout(Xtest,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    errq, rms, = tls.errors(Ytest, Y,["errq","rms"])
    errRMS3.append(rms)
    #print("\nApprentissage : (Erreur, RMS) = (%f, %f)" %(errq,rms));
    titre='pmc' + str(j)
    plt.plot(Xtrain,Yt, linewidth=1.5,label=titre);   
plt.legend(numpoints=1);
plt.subplot(1,2,2)   #ax=plt.gca()
#plt.figure();
plt.plot(Nneurone,errRMSt,'r*-',markersize=8);
plt.plot(Nneurone,errRMSv,'g*-',markersize=8);
plt.plot(Nneurone,errRMS3,'b*-',markersize=8); 
plt.legend(["RMS données d'apprentissage","RMS données validation","RMS données test"],numpoints=1);   
plt.title("RMS après apprentissage sur 1/4  \n de données  pour different nombre de neurones")
print(n, errRMSt)
print(n, errRMSv)
print(n, errRMS3)

n=5
Xtrain = Xi[0:-1:n]
Ytrain = Yi[0:-1:n]
Xval = Xi[1::n]
Yval = Yi[1::n]
errRMSt = [] 
errRMSv = []         
errRMS3 = []
plt.figure();
plt.subplot(1,2,1)   #ax=plt.gca()
plt.plot(Xtrain,Ytrain,'b*',label='Train', markersize=8);    # training datas
plt.plot(Xi,YC,'r',label='True', linewidth=1.5);    # validation datas
plt.title("Prédictions après apprentissage avec \n 1/5 de données pour différent nombre de neurones ", fontsize = 15)
#plt.legend(['train', 'true'], numpoints=1)
for j in range(1,10, 2):
    m  = [j]; # Une couche cachée de 5 neurones
#
    # Fonctions de Transfert : mettre autant de Fi que de m+1
    F1 = "tah"; F2 = "lin"; 
    FF = [F1,F2];
    # Les poids
    WW   = tdp.pmcinit(Xtrain, Ytrain, m);
    # > Learning
#    nbitemax = 2000;
#    dfreq    = 200;  
    nbitemax = 10000; #1000
    dfreq    = 10; #200 
    dcurves  = 0;
    WW, it_minval = tdp.pmctrain(Xtrain,Ytrain,WW,FF, \
     nbitemax=nbitemax, dfreq=dfreq, dcurves=dcurves,  Xv=Xval,Yv=Yval);
    Yt = tdp.pmcout(Xtrain,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    # Error et RMS en Apprentissage :
    errq, rmst, = tls.errors(Ytrain, Yt,["errq","rms"])
    errRMSt.append(rmst)
    Yv = tdp.pmcout(Xval,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    # Error et RMS en Apprentissage :
    errq, rmsv, = tls.errors(Yval, Yv,["errq","rms"])
    errRMSv.append(rmsv)
    Y = tdp.pmcout(Xtest,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    errq, rms, = tls.errors(Ytest, Y,["errq","rms"])
    errRMS3.append(rms)
    #print("\nApprentissage : (Erreur, RMS) = (%f, %f)" %(errq,rms));
    titre='pmc' + str(j)
    plt.plot(Xtrain,Yt, linewidth=1.5,label=titre);   
plt.legend(numpoints=1);
plt.subplot(1,2,2)   #ax=plt.gca()
#plt.figure();
plt.plot(Nneurone,errRMSt,'r*-',markersize=8);
plt.plot(Nneurone,errRMSv,'g*-',markersize=8);
plt.plot(Nneurone,errRMS3,'b*-',markersize=8); 
plt.legend(["RMS données d'apprentissage","RMS données validation","RMS données test"],numpoints=1);   
plt.title("RMS après apprentissage sur 1/5  \n de données  pour different nombre de neurones")
print(n, errRMSt)
print(n, errRMSv)
print(n, errRMS3)

n=10
Xtrain = Xi[0:-1:n]
Ytrain = Yi[0:-1:n]
Xval = Xi[1::n]
Yval = Yi[1::n]
errRMSt = [] 
errRMSv = []         
errRMS3 = []
plt.figure();
plt.subplot(1,2,1)   #ax=plt.gca()
plt.plot(Xtrain,Ytrain,'b*',label='Train', markersize=8);    # training datas
plt.plot(Xi,YC,'r',label='True', linewidth=1.5);    # validation datas
plt.title("Prédictions après apprentissage avec \n 1/10 de données pour différent nombre de neurones ", fontsize = 15)
#plt.legend(['train', 'true'], numpoints=1)
for j in range(1,10, 2):
    m  = [j]; # Une couche cachée de 5 neurones
#
    # Fonctions de Transfert : mettre autant de Fi que de m+1
    F1 = "tah"; F2 = "lin"; 
    FF = [F1,F2];
    # Les poids
    WW   = tdp.pmcinit(Xtrain, Ytrain, m);
    # > Learning
#    nbitemax = 2000;
#    dfreq    = 200;  
    nbitemax = 10000; #1000
    dfreq    = 10; #200 
    dcurves  = 0;
    WW, it_minval = tdp.pmctrain(Xtrain,Ytrain,WW,FF, \
     nbitemax=nbitemax, dfreq=dfreq, dcurves=dcurves,  Xv=Xval,Yv=Yval);
    Yt = tdp.pmcout(Xtrain,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    # Error et RMS en Apprentissage :
    errq, rmst, = tls.errors(Ytrain, Yt,["errq","rms"])
    errRMSt.append(rmst)
    Yv = tdp.pmcout(Xval,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    # Error et RMS en Apprentissage :
    errq, rmsv, = tls.errors(Yval, Yv,["errq","rms"])
    errRMSv.append(rmsv)
    Y = tdp.pmcout(Xtest,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    errq, rms, = tls.errors(Ytest, Y,["errq","rms"])
    errRMS3.append(rms)
    #print("\nApprentissage : (Erreur, RMS) = (%f, %f)" %(errq,rms));
    titre='pmc' + str(j)
    plt.plot(Xtrain,Yt, linewidth=1.5,label=titre);   
plt.legend(numpoints=1);
plt.subplot(1,2,2)   #ax=plt.gca()
#plt.figure();
plt.plot(Nneurone,errRMSt,'r*-',markersize=8);
plt.plot(Nneurone,errRMSv,'g*-',markersize=8);
plt.plot(Nneurone,errRMS3,'b*-',markersize=8); 
plt.legend(["RMS données d'apprentissage","RMS données validation","RMS données test"],numpoints=1);   
plt.title("RMS après apprentissage sur 1/10  \n de données  pour different nombre de neurones")
print(n, errRMSt)
print(n, errRMSv)
print(n, errRMS3)

n=30
Xtrain = Xi[0:-1:n]
Ytrain = Yi[0:-1:n]
Xval = Xi[1::n]
Yval = Yi[1::n]
errRMSt = [] 
errRMSv = []         
errRMS3 = []
plt.figure();
plt.subplot(1,2,1)   #ax=plt.gca()
plt.plot(Xtrain,Ytrain,'b*',label='Train', markersize=8);    # training datas
plt.plot(Xi,YC,'r',label='True', linewidth=1.5);    # validation datas
plt.title("Prédictions après apprentissage avec \n 1/30 de données pour différent nombre de neurones ", fontsize = 15)
#plt.legend(['train', 'true'], numpoints=1)
for j in range(1,10, 2):
    m  = [j]; # Une couche cachée de 5 neurones
#
    # Fonctions de Transfert : mettre autant de Fi que de m+1
    F1 = "tah"; F2 = "lin"; 
    FF = [F1,F2];
    # Les poids
    WW   = tdp.pmcinit(Xtrain, Ytrain, m);
    # > Learning
#    nbitemax = 2000;
#    dfreq    = 200;  
    nbitemax = 10000; #1000
    dfreq    = 10; #200 
    dcurves  = 0;
    WW, it_minval = tdp.pmctrain(Xtrain,Ytrain,WW,FF, \
     nbitemax=nbitemax, dfreq=dfreq, dcurves=dcurves,  Xv=Xval,Yv=Yval);
    Yt = tdp.pmcout(Xtrain,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    # Error et RMS en Apprentissage :
    errq, rmst, = tls.errors(Ytrain, Yt,["errq","rms"])
    errRMSt.append(rmst)
    Yv = tdp.pmcout(Xval,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    # Error et RMS en Apprentissage :
    errq, rmsv, = tls.errors(Yval, Yv,["errq","rms"])
    errRMSv.append(rmsv)
    Y = tdp.pmcout(Xtest,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    errq, rms, = tls.errors(Ytest, Y,["errq","rms"])
    errRMS3.append(rms)
    #print("\nApprentissage : (Erreur, RMS) = (%f, %f)" %(errq,rms));
    titre='pmc' + str(j)
    plt.plot(Xtrain,Yt, linewidth=1.5,label=titre);   
plt.legend(numpoints=1);
plt.subplot(1,2,2)   #ax=plt.gca()
#plt.figure();
plt.plot(Nneurone,errRMSt,'r*-',markersize=8);
plt.plot(Nneurone,errRMSv,'g*-',markersize=8);
plt.plot(Nneurone,errRMS3,'b*-',markersize=8); 
plt.legend(["RMS données d'apprentissage","RMS données validation","RMS données test"],numpoints=1);   
plt.title("RMS après apprentissage sur 1/30  \n de données  pour different nombre de neurones")
print(n, errRMSt)
print(n, errRMSv)
print(n, errRMS3)

n=100
Xtrain = Xi[0:-1:n]
Ytrain = Yi[0:-1:n]
Xval = Xi[1::n]
Yval = Yi[1::n]
errRMSt = [] 
errRMSv = []         
errRMS3 = []
plt.figure();
plt.subplot(1,2,1)   #ax=plt.gca()
plt.plot(Xtrain,Ytrain,'b*',label='Train', markersize=8);    # training datas
plt.plot(Xi,YC,'r',label='True', linewidth=1.5);    # validation datas
plt.title("Prédictions après apprentissage avec \n 1/100 de données pour différent nombre de neurones ", fontsize = 15)
#plt.legend(['train', 'true'], numpoints=1)
for j in range(1,10, 2):
    m  = [j]; # Une couche cachée de 5 neurones
#
    # Fonctions de Transfert : mettre autant de Fi que de m+1
    F1 = "tah"; F2 = "lin"; 
    FF = [F1,F2];
    # Les poids
    WW   = tdp.pmcinit(Xtrain, Ytrain, m);
    # > Learning
#    nbitemax = 2000;
#    dfreq    = 200;  
    nbitemax = 10000; #1000
    dfreq    = 10; #200 
    dcurves  = 0;
    WW, it_minval = tdp.pmctrain(Xtrain,Ytrain,WW,FF, \
     nbitemax=nbitemax, dfreq=dfreq, dcurves=dcurves,  Xv=Xval,Yv=Yval);
    Yt = tdp.pmcout(Xtrain,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    # Error et RMS en Apprentissage :
    errq, rmst, = tls.errors(Ytrain, Yt,["errq","rms"])
    errRMSt.append(rmst)
    Yv = tdp.pmcout(Xval,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    # Error et RMS en Apprentissage :
    errq, rmsv, = tls.errors(Yval, Yv,["errq","rms"])
    errRMSv.append(rmsv)
    Y = tdp.pmcout(Xtest,WW,FF); # Valeur de sortie du PMC sur l'ensemble d'App.
    errq, rms, = tls.errors(Ytest, Y,["errq","rms"])
    errRMS3.append(rms)
    #print("\nApprentissage : (Erreur, RMS) = (%f, %f)" %(errq,rms));
    titre='pmc' + str(j)
    plt.plot(Xtrain,Yt, linewidth=1.5,label=titre);   
plt.legend(numpoints=1);
plt.subplot(1,2,2)   #ax=plt.gca()
#plt.figure();
plt.plot(Nneurone,errRMSt,'r*-',markersize=8);
plt.plot(Nneurone,errRMSv,'g*-',markersize=8);
plt.plot(Nneurone,errRMS3,'b*-',markersize=8); 
plt.legend(["Erreur données d'apprentissage","Erreur données validation","Erreur données test"],numpoints=1);   
plt.title("Erreurs après apprentissage sur 1/100  \n de données  pour different nombre de neurones")
print(n, errRMSt)
print(n, errRMSv)
print(n, errRMS3)