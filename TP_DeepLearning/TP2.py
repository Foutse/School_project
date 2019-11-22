# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 04:17:22 2018

@author: FOUTSE
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation,Flatten
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D

from keras.datasets import mnist
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
#############################################################
mpl.use('TKAgg')
plt.figure(figsize=(7.195, 3.841), dpi=100)
for i in range(200):
    plt.subplot(10,20,i+1)
    plt.imshow(X_train[i,:].reshape([28,28]), cmap='gray')
    plt.axis('off')
plt.show()
##############################################################################################
#Exo1: Regression Logistic avec Keras#########################################################

################################################################################################
model = Sequential() #On créé ainsi un réseau de neurones vide                                ##
model.add(Dense(10,  input_dim=784, name='fc1')) #l’ajout d’une couche de projection linéaire ##
                                                 #(couche complètement connectée) de taille 10##
model.add(Activation('softmax')) #l’ajout d’une couche d’activation de type softmax           ##
model.summary()                                                                               ##
################################################################################################
#Question :
#Quel modèle de prédiction reconnaissez-vous ? Vérifier le nombre de paramètres du réseau 
#à apprendre dans la méthode summary(). - Écrire un script exo1.py permettant de créer le 
#réseau de neurone ci-dessus.
###############################################################################################
learning_rate = 0.1                                                                          ##
sgd = SGD(learning_rate) #méthode d’optimisation (descente de gradient stochastique)         ##
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])            ##
#on compiler le modèle en lui passant un loss (l’ entropie croisée),                         ##
#une méthode d’optimisation (sgd), et une métrique d’évaluation( le taux de bonne prédiction ## 
#des catégories, accuracy):                                                                  ##
###############################################################################################
batch_size = 100 #nombre d’exemples utilisé pour estimer le gradient de la fonction de coût. ##
nb_epoch = 20 # nombre de passages sur l’ensemble des exemples de la base d’apprentissage)   ##
             #lors de la descente de gradient.                                               ##
K=10                                                                                         ##
###############################################################################################
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, K)
Y_test = np_utils.to_categorical(y_test, K)
model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1)
#Apprentissage du modele avec la methode fit
#Le premier élément de score renvoie la fonction de coût sur la base de test, 
#le second élément renvoie le taux de bonne détection (accuracy).
scores = model.evaluate(X_test, Y_test, verbose=0) #evaluation du modele sur l'ensemble de test
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
###############################################################################################
#Implémenter l’apprentissage du modèle sur la base de train de la base MNIST.
#Évaluer les performances du réseau sur la base de test et les comparer à celles obtenues lors 
#de la séance précédente (ré-implémentation manuelle de l’algorithme de rétro-propagation). Conclure.
#Obtained the following results
#loss: 27.06%
#acc: 92.34%


#Exercice 2 : Perceptron avec Keras
#On va maintenant enrichir le modèle de régression logistique en créant une couche de neurones cachée 
#complètement connectée supplémentaire, suivie d’une fonction d’activation non linéaire de type sigmoïde. 
#On va ainsi obtenir un réseau de neurones à une couche cachée, le Perceptron (cf séance précédente)
###############################################################################################
model = Sequential()
model.add(Dense(100,  input_dim=784, name='fc2'))
model.add(Activation('sigmoid'))
model.add(Dense(10, name='fc3'))
model.add(Activation('softmax'))
model.summary()
##############################################################
learning_rate = 1.0
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
################################################################################
batch_size = 100
nb_epoch = 100
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1)
#################################################################################
scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
################################################################################
from keras.models import model_from_yaml
def saveModel(model, savename):
  # serialize model to YAML
  model_yaml = model.to_yaml()
  with open(savename+".yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
    print("Yaml Model ",savename,".yaml saved to disk")
  # serialize weights to HDF5
  model.save_weights(savename+".h5")
  print("Weights ",savename,".h5 saved to disk")
  
saveModel(model, 'my_model_MLP')
####Results obtained#############################################################
#loss: 9.05%
#acc: 97.95%





#Exercice 3 : Réseaux de neurones convolutifs avec Keras
#On va maintenant étendre le perceptron de l’exercice précédent pour mettre 
#en place un réseau de neurones convolutif profond, “Convolutionnal Neural Networks”, ConvNets.
#
#Les réseaux convolutifs manipulent des images multi-dimensionnelles en entrée (tenseurs). 
#On va donc commencer par reformater les données d’entrée afin que chaque exemple soit de taille 28×28×1
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

model = Sequential()

conv1= Conv2D(32,kernel_size=(5, 5),activation='relu',input_shape=(28, 28, 1),padding='valid')
#32 est le nombre de filtres
#(5, 5) est la taille spatiale de chaque filtre (masque de convolution).
#padding=’valid’ correspond ignorer les bords lors du calcul (et donc à diminuer la taille 
#spatiale en sortie de la convolution).
#avec une couche de convolution la non-linéarité en sortie de la convolution, 
#comme illustré ici dans l’exemple avec une fonction d’activation de type relu.
pool1 = MaxPooling2D(pool_size=(2, 2)) #déclaration d'une couche de max-pooling
#Des couches d’agrégation spatiale (pooling), afin de permettre une invariance aux translations locales.
#(2, 2) est la taille spatiale sur laquelle l’opération d’agrégation est effectuée. 
#on obtient donc des cartes de sorties avec des tailles spatiales divisées par deux 
#par rapport à la taille d’entrée

model.add(conv1)
#L'ajout d'une couche de convolution avec 16 filtres de taille 5×5, suivie d’une non linéarité de type relu 
model.add(pool1)
#Ajout d’une couche de max pooling de taille 2×2.

conv2= Conv2D(16,kernel_size=(5, 5),activation='relu',input_shape=(28, 28, 1),padding='valid')
pool2 = MaxPooling2D(pool_size=(2, 2))
model.add(conv2)
#L'ajout d'une seconde couche de convolution avec 16 filtres de taille 5×5, 
#suivie d’une non linéarité de type relu 
model.add(pool2)
#puis d’une seconde couche de max pooling de taille 2×2.
model.add(Flatten()) #On mait a plat les couches convolutives précédentes
#Comme dans le réseau LeNet, on considérera la sortie du second bloc convolutif comme un vecteur, 
#ce que revient à “mettre à plat” les couches convolutives précédentes (model.add(Flatten())).
model.add(Dense(100,  input_dim=784, name='fc4'))
#L'ajout d'une couche complètement connectée de taille 100, 
model.add(Activation('sigmoid'))
#suivie d’une non linéarité de type sigmoïde.
model.add(Dense(10, name='fc5'))
#Une couche complètement connectée de taille 10, 
model.add(Activation('softmax'))
#suivie d’une non linéarité de type softmax.
model.summary()

############################################################
learning_rate = 1.0
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
################################################################################
batch_size = 100
nb_epoch = 50
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1)
#################################################################################
scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
################################################################################
#Apprendre le modèle et évaluer les performances du réseau sur la base de test. 
#Vous devez obtenir un score de l’ordre de 99% pour ce modèle ConvNet.
#Quelle est le temps d’une époque avec ce modèle convolutif ? Ans: 38s 641us/step
#On pourra sauvegarder le modèle appris avec la méthode saveModel précédente
saveModel(model, 'my_model_ConvNet')
#resultats
#loss: 2.91%
#acc: 99.20%


#Exercice 4 : Visualisation avec t-SNE##########################################################
#On va appliquer la méthode t-SNE sur les données brutes de la base de test de MNIST 
#en utilisant la classe TSNE du module sklearn.manifold
#l’objectif est d’effectuer une réduction de dimension en 2D des données de la base de test de MNIST 
#en utilisant la méthode t-SNE


import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import ConvexHull
from sklearn.mixture import GaussianMixture
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

###########################################################
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#X_= TSNE(n_components=2, perplexity=30.0,
#        early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, 
#        n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', 
#        init='random', verbose=0, random_state=None, method='barnes_hut', angle=0.5)

X_embedded = TSNE(n_components=2, perplexity=30.0, init='pca', verbose=2).fit_transform(X_test)

X_embedded_PCA = PCA(n_components=2, svd_solver='full').fit_transform(X_test)

#################################################################################
def convexHulls(points, labels):
  # computing convex hulls for a set of points with asscoiated labels
  convex_hulls = []
  for i in range(10):
    convex_hulls.append(ConvexHull(points[labels==i,:]))
  return convex_hulls
# Function Call
convex_hulls= convexHulls(X_embedded, y_test)
convex_hullsPca= convexHulls(X_embedded_PCA, y_test)

####################################################################################

def best_ellipses(points, labels):
  # computing best fiiting ellipse for a set of points with asscoiated labels
  gaussians = []
  for i in range(10):
    gaussians.append(GaussianMixture(n_components=1, covariance_type='full').fit(points[labels==i, :]))
  return gaussians
# Function Call
ellipses = best_ellipses(X_embedded, y_test)
ellipsesPca = best_ellipses(X_embedded_PCA, y_test)

def neighboring_hit(points, labels):
  k = 6
  nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
  distances, indices = nbrs.kneighbors(points)

  txs = 0.0
  txsc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  nppts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

  for i in range(len(points)):
    tx = 0.0
    for j in range(1,k+1):
      if (labels[indices[i,j]]== labels[i]):
        tx += 1
        tx /= k
        txsc[labels[i]] += tx
        nppts[labels[i]] += 1
        txs += tx

  for i in range(10):
    txsc[i] /= nppts[i]

  return txs / len(points)

nh= neighboring_hit(X_embedded, y_test)
nhPca= neighboring_hit(X_embedded_PCA, y_test)


def visualization(points2D, labels, convex_hulls, ellipses ,projname, nh):

  points2D_c= []
  for i in range(10):
      points2D_c.append(points2D[labels==i, :])
  # Data Visualization
  cmap =cm.tab10

  plt.figure(figsize=(3.841, 7.195), dpi=100)
  plt.set_cmap(cmap)
  plt.subplots_adjust(hspace=0.4 )
  plt.subplot(311)
  plt.scatter(points2D[:,0], points2D[:,1], c=labels,  s=3,edgecolors='none', cmap=cmap, alpha=1.0)
  plt.colorbar(ticks=range(10))

  plt.title("2D "+projname+" - NH="+str(nh*100.0))

  vals = [ i/10.0 for i in range(10)]
  sp2 = plt.subplot(312)
  for i in range(10):
      ch = np.append(convex_hulls[i].vertices,convex_hulls[i].vertices[0])
      sp2.plot(points2D_c[i][ch, 0], points2D_c[i][ch, 1], '-',label='$%i$'%i, color=cmap(vals[i]))
  plt.colorbar(ticks=range(10))
  plt.title(projname+" Convex Hulls")

  def plot_results(X, Y_, means, covariances, index, title, color):
      splot = plt.subplot(3, 1, 3)
      for i, (mean, covar) in enumerate(zip(means, covariances)):
          v, w = linalg.eigh(covar)
          v = 2. * np.sqrt(2.) * np.sqrt(v)
          u = w[0] / linalg.norm(w[0])
          # as the DP will not use every component it has access to
          # unless it needs it, we shouldn't plot the redundant
          # components.
          if not np.any(Y_ == i):
              continue
          plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color, alpha = 0.2)

          # Plot an ellipse to show the Gaussian component
          angle = np.arctan(u[1] / u[0])
          angle = 180. * angle / np.pi  # convert to degrees
          ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
          ell.set_clip_box(splot.bbox)
          ell.set_alpha(0.6)
          splot.add_artist(ell)

      plt.title(title)
  plt.subplot(313)

  for i in range(10):
      plot_results(points2D[labels==i, :], ellipses[i].predict(points2D[labels==i, :]), ellipses[i].means_,
      ellipses[i].covariances_, 0,projname+" fitting ellipses", cmap(vals[i]))

  plt.savefig(projname+".png", dpi=100)
  plt.show()
  

#[t-SNE] Error after 1000 iterations: 1.781618
visualization(X_embedded, y_test, convex_hulls, ellipses ,'t-SNE', nh)
visualization(X_embedded_PCA, y_test, convex_hullsPca, ellipsesPca ,'PCA', nhPca)


#Exercice 5 : Visualisation des représentations internes des réseaux de neurones##############################
##############################################################################################################
#On va maintenant s’intéresser à visualisation de l’effet de “manifold untangling” permis par les réseaux  ###
#de neurones.                                                                                              ###
#l’objectif va être d’utiliser la méthode t-SNE de l’exercice 2 pour projeter les couches cachés des réseaux##
#de neurones dans un espace de dimension 2, ce qui permettra de visualiser la distribution des              ##
#représentations internes et des labels.                                                                    ##
#

###########################################################

def loadModel(savename):
 with open(savename+".yaml", "r") as yaml_file:
  model = model_from_yaml(yaml_file.read())
 print( "Yaml Model ",savename,".yaml loaded ")
 model.load_weights(savename+".h5")
 print( "Weights ",savename,".h5 loaded ")
 return model

# LOADING MODEL
nameModel = "my_model_MLP" #REPLACE WITH YOUR MODEL NAME
model = loadModel(nameModel)
model.summary()

# convert class vectors to binary class matrices
Y_test = np_utils.to_categorical(y_test, 10)
# COMPILING MODEL
learning_rate = 1.0
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

scores_test = model.evaluate(X_test, Y_test, verbose=1)
print("PERFS TEST: %s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100))
#On évalue les performances du modèle chargé sur la base de test de MNIST pour vérifier son comportement.
#PERFS TEST: acc: 97.97%
model.pop() #permettant de supprimer la couche au sommet du modèle
#On vas l'appliquer deux fois (on supprime la couche d’activation softmax et la couche complètement connectée)

model.summary()
model.pop()
model.summary()
predict = model.predict(X_test)
######################################################################################
#Ensuite on va utiliser la méthode t-SNE mise en place à l’exercice 2 pour visualiser 
#les représentations internes des données.
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

###########################################################
X_embedded = TSNE(n_components=2, perplexity=30.0, init='pca', verbose=2).fit_transform(predict)
#[t-SNE] Error after 1000 iterations: 1.328868
#X_embedded_PCA = PCA(n_components=2, svd_solver='full').fit_transform(X_train_5000)
# Function Call
convex_hulls= convexHulls(X_embedded, y_test)
####################################################################################
# Function Call
ellipses = best_ellipses(X_embedded, y_test)
nh= neighboring_hit(X_embedded, y_test)

visualization(X_embedded, y_test, convex_hulls, ellipses ,'t-SNE_MLP', nh)


##Pour le model ConvNet my_model_ConvNet
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# LOADING MODEL
nomModel = "my_model_ConvNet"
model = loadModel(nomModel)
model.summary()

# convert class vectors to binary class matrices
Y_test = np_utils.to_categorical(y_test, 10)
# COMPILING MODEL
learning_rate = 1.0
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

scores_test = model.evaluate(X_test, Y_test, verbose=1)
print("PERFS TEST: %s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100))
#On évalue les performances du modèle chargé sur la base de test de MNIST pour vérifier son comportement.
#PERFS TEST: acc: 99.20%
model.pop() #permettant de supprimer la couche au sommet du modèle
#On vas l'appliquer deux fois (on supprime la couche d’activation softmax et la couche complètement connectée)

model.summary()
model.pop()
model.summary()
predict = model.predict(X_test)
######################################################################################
#Ensuite on va utiliser la méthode t-SNE mise en place à l’exercice 2 pour visualiser 
#les représentations internes des données.
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

###########################################################
X_embedded = TSNE(n_components=2, perplexity=30.0, init='pca', verbose=2).fit_transform(predict)
#[t-SNE] Error after 1000 iterations: 1.329938
#X_embedded_PCA = PCA(n_components=2, svd_solver='full').fit_transform(X_train_5000)
# Function Call
convex_hulls= convexHulls(X_embedded, y_test)
####################################################################################
# Function Call
ellipses = best_ellipses(X_embedded, y_test)
nh= neighboring_hit(X_embedded, y_test)

visualization(X_embedded, y_test, convex_hulls, ellipses ,'t-SNE_CNN', nh)