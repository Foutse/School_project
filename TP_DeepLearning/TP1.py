from keras.datasets import mnist
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.utils import np_utils
import numpy as np

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
print(X_train.shape[1], 'train samples')
N=60000;

mpl.use('TKAgg')
plt.figure(figsize=(7.195, 3.841), dpi=100)
for i in range(200):
  plt.subplot(10,20,i+1)
  plt.imshow(X_train[i,:].reshape([28,28]), cmap='gray')
  plt.axis('off')
plt.show()

#Quel est l’espace dans lequel se trouvent les images ? Quel est sa taille ?

#Quel est le nombre de paramètres du modèle ? Justifier le calcul. 
#le nombre de parametres (784*10 + 1)

#La fonction de coût de l’Eq. (3) est-elle convexe par rapports aux paramètres W, b du modèle ? Avec un pas de gradient bien choisi, peut-on assurer la convergence vers le minimum global de la solution ?



K=10
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, K)
Y_test = np_utils.to_categorical(y_test, K)

d = X_train.shape[1]
W = np.zeros((d,K))
b = np.zeros((1,K))
numEp = 20 # Number of epochs for gradient descent
eta = 1e-1 # Learning rate
batch_size = 100
nb_batches = int(float(N) / batch_size)
gradW = np.zeros((d,K))
gradb = np.zeros((1,K))

#print("1")
def forward1(batch, W, b):
    #y_bar = np.dot(X_train,W);
    numerateur = np.exp(np.dot(batch,W)+b);
    denominateur = np.sum(numerateur, axis=1)[:,None]
    y_bar = numerateur/denominateur;
    return y_bar;

def accuracy(W, b, images, labels):
  pred = forward1(images, W,b )
  return np.where( pred.argmax(axis=1) != labels.argmax(axis=1) , 0.,1.).mean()*100.0


#print("2")
for epoch in range(numEp):
  for ex in range(nb_batches):
     # FORWARD PASS : compute prediction with current params for examples in batch
     batch = X_train[ex*batch_size:(ex+1)*batch_size,:]
     y_batch = Y_train[ex*batch_size:(ex+1)*batch_size,:]
     Y_bar = forward1(batch,W,b);
     # BACKWARD PASS :
     # 1) compute gradients for W and b
     gradW = (1/batch_size)*np.dot(batch.T,(-y_batch+Y_bar))
     gradb = (1/batch_size)*np.sum((-y_batch+Y_bar),0)
     # 2) update W and b parameters with gradient descent
     W = W - eta*gradW
     b = b - eta*gradb
  #print("{} , {}".format(gradb,accuracy(W, b, batch, Y_train[ex*batch_size:(ex+1)*batch_size,:])))  
#print("3")
     
print("{}".format(accuracy(W, b, X_test, Y_test)))

#%%
#Exercice 2
K=10
L = 100
Wh = np.zeros((d,L))
bh = np.zeros((1,L))
Wy = np.zeros((L,K))
by = np.zeros((1,K))
#gradWy = np.zeros((L,K))
#gradby = np.zeros((1,K))
#gradv = np.zeros((batch_size,K))
#gradWh = np.zeros((d,L))
#gradbh = np.zeros((1,L))
numEp = 100 # Number of epochs for gradient descent
eta = 1.0 # Learning rate

#print("1")
def softmax(X):
 # Input matrix X of size Nbxd - Output matrix of same size
 E = np.exp(X)
 return (E.T / np.sum(E,axis=1)).T

def forward2(batch, Wh, bh, Wy, by):
    #y_bar = np.dot(X_train,W);
    u = np.dot(batch,Wh)+bh
    h= 1/(1+np.exp(-1*u));
    v = np.dot(h,Wy)+by
    y_bar = softmax(v)
    return y_bar, h;

def accuracy2(Wh, bh, Wy, by, images, labels):
  pred, Hn = forward2(images, Wh, bh, Wy, by )
  return np.where( pred.argmax(axis=1) != labels.argmax(axis=1) , 0.,1.).mean()*100.0


#print("2")
for epoch in range(numEp):
  for ex in range(nb_batches):
     # FORWARD PASS : compute prediction with current params for examples in batch
     batch = X_train[ex*batch_size:(ex+1)*batch_size,:]
     y_batch = Y_train[ex*batch_size:(ex+1)*batch_size,:]
     Y_bar, H = forward2(batch, Wh, bh, Wy, by);
     # BACKWARD PASS :
     # 1) compute gradients for W and b
     gradv = Y_bar - y_batch
     gradWy = (1/batch_size)*np.dot(H.T,(-y_batch+Y_bar))
     gradby = (1/batch_size)*np.sum((-y_batch+Y_bar),axis=0)
     Wy = Wy - eta*gradWy
     by = by - eta*gradby
     tmp1 = np.zeros((batch_size,L))
     tmp2 = np.zeros(np.shape(H))
     delta_h = np.zeros((batch_size,L))
     for i in range(batch_size):
         tmp1 = np.dot(gradv[i,:],(gradWy.T))
         tmp2[i,:] = tmp1*H[i,:]*(1-H[i,:])
     #delta_h[i,:] = tmp2
     gradWh = (1/batch_size)*np.dot(batch.T,delta_h)
     gradbh = (1/batch_size)*np.sum((delta_h),axis=0)
     # 2) update W and b parameters with gradient descent  
     Wh = Wh - eta*gradWh
     bh = bh - eta*gradbh
     
  #print("{} , {}".format(gradb,accuracy(W, b, batch, Y_train[ex*batch_size:(ex+1)*batch_size,:])))  
#print("3")
     
print("{}".format(accuracy2(Wh, bh, Wy, by, X_test, Y_test)))