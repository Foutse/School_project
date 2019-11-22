# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 13:43:33 2019
"""

import numpy as np
import pandas as pd
import _pickle as pickle
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, RMSprop
from sklearn.cluster import KMeans
from keras.models import model_from_yaml
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm


bStart = False
fin = open("fleurs_mal.txt", 'r', encoding="utf8")
lines = fin.readlines()
lines2 = []
text = []

for line in lines:
 line = line.strip().lower() # Remove blanks and capitals
 if("Charles Baudelaire avait un ami".lower() in line and bStart==False):
  print("START")
  bStart = True
 if("End of the Project Gutenberg EBook of Les Fleurs du Mal, by Charles Baudelaire".lower() in line):
  print("END")
  break
 if(bStart==False or len(line) == 0):
  continue
 lines2.append(line)

fin.close()
text = " ".join(lines2)
chars = sorted(set([c for c in text]))
nb_chars = len(chars)

SEQLEN = 10 # Length of the sequence to predict next char
STEP = 1 # stride between two subsequent sequences
input_chars = []
label_chars = []
for i in range(0, len(text) - SEQLEN, STEP):
       input_chars.append(text[i:i + SEQLEN])
       label_chars.append(text[i + SEQLEN])
nbex = len(input_chars)

"""
Question : Comment s’interprète la variable chars ? Que représente nb_chars ?

Reponse:
La variable chars est un dictionnaire regroupant et classant par odre croissant 
l'ensemble des caractères (distincts) présents dans le texte.

nb_chars représente tout simplement la taille du dictionnaire chars c'est à le 
nombre de caractères distincts présent dans le texte.
"""


#Ici on cree des labels artificiellement on prend 10 lettres puis la lettres suivante est sont label,
#ainsi on construit nos inputs et labels
#pp=text[0:0 + SEQLEN]
#ppl = text[0 + SEQLEN]

#On va maintenant vectoriser les données d’entraînement en utilisant le dictionnaire et un encodage 
#one-hot pour chaque caractère.

# mapping char -> index in dictionary: used for encoding (here)
char2index = dict((c, i) for i, c in enumerate(chars))
# mapping char -> index in dictionary: used for decoding, i.e. generation - part c)
index2char = dict((i, c) for i, c in enumerate(chars)) # mapping index -> char in dictionary

X = np.zeros((len(input_chars), SEQLEN, nb_chars), dtype=np.bool)
y = np.zeros((len(input_chars), nb_chars), dtype=np.bool)

for i, input_char in enumerate(input_chars):
 for j, ch in enumerate(input_char):
     # Fill X at correct index
     l = char2index.get(ch)
     #print(l)
     X[i,j,l] = True;
     # Fill y at correct index
     l = char2index[label_chars[i]]
     y[i,l] = True;

#char2index & index2char c'est le dictionnaire des symboles
#Chaque séquence d’entraînement est donc représentée par une matrice de taille SEQLEN×tdict, 
#correspondant à une longueur de SEQLEN caractères, chaque caratère étant encodé par un vecteur 
#binaire correspondant à un encodage one-hot.
ratio_train = 0.8
nb_train = int(round(len(input_chars)*ratio_train))
print("nb tot=",len(input_chars) , "nb_train=",nb_train)
X_train = X[0:nb_train,:,:]
y_train = y[0:nb_train,:]

X_test = X[nb_train:,:,:]
y_test = y[nb_train:,:]
print("X train.shape=",X_train.shape)
print("y train.shape=",y_train.shape)

print("X test.shape=",X_test.shape)
print("y test.shape=",y_test.shape)

outfile = "Baudelaire_len_"+str(SEQLEN)+".p"

with open(outfile, "wb" ) as pickle_f:
 pickle.dump( [index2char, X_train, y_train, X_test, y_test], pickle_f)


#Apprentissage

#SEQLEN = 10
#outfile = "Baudelaire_len_"+str(SEQLEN)+".p"
#[index2char, X_train, y_train, X_test, y_test] = pickle.load( open( outfile, "rb" ) )

##Pour optimiser des réseaux récurrents, on utilise préférentiellement des méthodes adaptatives comme 
##RMSprop [TH12]. On pourra donc compiler le modèle et utiliser la méthode summary() pour visualiser 
##le nombre de paramètres du réseaux
 
model = Sequential()
HSIZE = 128
model.add(SimpleRNN(HSIZE, return_sequences=False, input_shape=(SEQLEN, nb_chars),unroll=True))
model.add(Dense(nb_chars))
model.add(Activation("softmax"))
BATCH_SIZE = 128
NUM_EPOCHS = 50
learning_rate = 0.001
optim = RMSprop(lr=learning_rate)
model.compile(loss="categorical_crossentropy", optimizer=optim,metrics=['accuracy'])
model.summary()

"""
Question : Expliquer à quoi correspond return_sequences=False. 
N.B. : unroll=True permettra simplement d’accélérer les calculs.

Reponse:
return_sequences=False indique que l'on ne souhaite pas renvoyer la dernière sortie
de la séquence de sortie ou la séquence complète

"""

model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

scores_train = model.evaluate(X_train, y_train, verbose=1)
scores_test = model.evaluate(X_test, y_test, verbose=1)
print("PERFS TRAIN: %s: %.2f%%" % (model.metrics_names[1], scores_train[1]*100))
print("PERFS TEST: %s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100))



def saveModel(model, savename):
  # serialize model to YAML
  model_yaml = model.to_yaml()
  with open(savename+".yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
    print ("Yaml Model ",savename,".yaml saved to disk")
  # serialize weights to HDF5
  model.save_weights(savename+".h5")
  print ("Weights ",savename,".h5 saved to disk")

def loadModel(savename):
  with open(savename+".yaml", "r") as yaml_file:
    model = model_from_yaml(yaml_file.read())
  print ("Yaml Model ",savename,".yaml loaded ")
  model.load_weights(savename+".h5")
  print ("Weights ",savename,".h5 loaded ")
  return model

saveModel(model,'model1Tp4')

SEQLEN = 10 
outfile = "Baudelaire_len_"+str(SEQLEN)+".p"
[index2char, X_train, y_train, X_test, y_test] = pickle.load( open( outfile, "rb" ) )

nameModel='model1Tp4';

model = loadModel(nameModel)
model.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
model.summary()
nb_chars = len(index2char)

seed =15608
char_init = ""
for i in range(SEQLEN):
 char = index2char[np.argmax(X_train[seed,i,:])]
 char_init += char
print("CHAR INIT: "+char_init)

test = np.zeros((1, SEQLEN, nb_chars), dtype=np.bool)
test[0,:,:] = X_train[seed,:,:]

def sampling(preds, temperature=1.0):
 preds = np.asarray(preds).astype('float64')
 predsN = pow(preds,1.0/temperature)
 predsN /= np.sum(predsN)
 probas = np.random.multinomial(1, predsN, 1)
 return np.argmax(probas)

nbgen = 400 # number of characters to generate (1,nb_chars)
gen_char = char_init
temperature  = 0.01

for i in range(nbgen):
 preds = model.predict(test)[0]  # shape (1,nb_chars)
 next_ind = sampling(preds,temperature)
 next_char = index2char[next_ind]
 gen_char += next_char
 for i in range(SEQLEN-1):
  test[0,i,:] = test[0,i+1,:]
 test[0,SEQLEN-1,:] = 0
 test[0,SEQLEN-1,next_ind] = 1

print("Generated text: "+gen_char)


filename = 'flickr_8k_train_dataset.txt'
df = pd.read_csv(filename, delimiter='\t')
nb_samples = df.shape[0]
iter = df.iterrows()
allwords = []
for i in range(nb_samples):
 x = iter.__next__()
 cap_words = x[1][1].split() # split caption into words
 cap_wordsl = [w.lower() for w in cap_words] # remove capital letters
 allwords.extend(cap_wordsl)

unique = list(set(allwords)) # List of different words in captions
print(len(unique))

GLOVE_MODEL = "glove.6B.100d.txt"
fglove = open(GLOVE_MODEL, "r", encoding="utf8")

listwords=[]
listembeddings=[]
cpt=0
for line in fglove:
 row = line.strip().split()
 word = row[0]#COMPLETE WITH YOUR CODE
 if(word in unique or word=='unk'):
  listwords.append(word)
  embedding = np.array(row[1:],dtype="float32")#COMPLETE WITH YOUR CODE - use a numpy array with dtype="float32"
  listembeddings.append(embedding)

  cpt +=1
  print("word: "+word+" embedded "+str(cpt))

fglove.close()
nbwords = len(listembeddings)
tembedding = len(listembeddings[0])
print("Number of words="+str(len(listembeddings))+" Embedding size="+str(tembedding))

embeddings = np.zeros((len(listembeddings)+2,tembedding+2))
for i in range(nbwords):
 embeddings[i,0:tembedding] = listembeddings[i]

"""
Question :
Expliquer l’objectif de la normalisation

"""
listwords.append('<start>')
embeddings[7001,100] = 1
listwords.append('<end>')
embeddings[7002,101] = 1

outfile = 'Caption_Embeddings.p'
with open(outfile, "wb" ) as pickle_f:
 pickle.dump( [listwords, embeddings], pickle_f)


outfile = 'Caption_Embeddings.p'
[listwords, embeddings] = pickle.load( open( outfile, "rb" ) )
print("embeddings: "+str(embeddings.shape))

for i in range(embeddings.shape[0]):
 embeddings[i,:] /= np.linalg.norm(embeddings[i,:])

kmeans = KMeans(n_clusters=10, max_iter=1000, init="random").fit(embeddings)# COMPLETE WITH YOUR CODE - apply fit() method on embeddings
clustersID  = kmeans.labels_
clusters = kmeans.cluster_centers_

indclusters=np.ndarray((10,7003),dtype=int)
for i in range(10):
 norm = np.linalg.norm((clusters[i] - embeddings),axis=1)
 inorms = np.argsort(norm)
 indclusters[i][:] = inorms[:]

 print("Cluster "+str(i)+" ="+listwords[indclusters[i][0]])
 for j in range(1,21):
  print(" mot: "+listwords[indclusters[i][j]])

tsne = TSNE(n_components=2, perplexity=30, verbose=2, init='pca', early_exaggeration=24)
points2D = tsne.fit_transform(embeddings)

pointsclusters = np.ndarray((10,2),dtype=int)
for i in range(10):
 pointsclusters[i,:] = points2D[int(indclusters[i][0])]

cmap =cm.tab10
plt.figure(figsize=(3.841, 7.195), dpi=100)
plt.set_cmap(cmap)
plt.subplots_adjust(hspace=0.4 )
plt.scatter(points2D[:,0], points2D[:,1], c=clustersID,  s=3,edgecolors='none', cmap=cmap, alpha=1.0)
plt.scatter(pointsclusters[:,0], pointsclusters[:,1], c=range(10),marker = '+', s=1000, edgecolors='none', cmap=cmap, alpha=1.0)

plt.colorbar(ticks=range(10))
plt.show()