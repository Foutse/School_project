# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 12:30:16 2019

@author: LENOVO
"""

import pandas as pd
import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_yaml
from keras.models import Sequential
from keras.layers import Dense, Activation, Masking, SimpleRNN
from keras.optimizers import Adam
#from keras.utils import np_utils
#from keras.optimizers import RMSprop
import nltk

filename = 'flickr_8k_test_dataset.txt'
df = pd.read_csv(filename, delimiter='\t')

nbkeep = 1000 # 100 is needed for fast processing

outfile = 'Caption_Embeddings_1000.p'
[listwords, embeddings] = pickle.load( open( outfile, "rb" ) )


nbTest = df.shape[0]
iter = df.iterrows()

caps = [] # Set of captions
imgs = [] # Set of images
for i in range(nbTest):
 x = iter.__next__()
 caps.append(x[1][1])
 imgs.append(x[1][0])


maxLCap =35
indexwords = {} # Useful for tensor filling
for i in range(len(listwords)):
 indexwords[listwords[i]] = i

# Loading images features
encoded_images = pickle.load( open( "encoded_images_PCA.p", "rb" ) )

# Allocating data and labels tensors
tinput = 202
tVocabulary = len(listwords)
X_test = np.zeros((nbTest,maxLCap, tinput))
Y_test = np.zeros((nbTest,maxLCap, tVocabulary), bool)

for i in range(nbTest):
 words_in_caption =  caps[i].split()
 indseq=0 # current sequence index (to handle mising words in reduced dictionary)
 for j in range(len(words_in_caption)-1):
  current_w = words_in_caption[j].lower()
  if(current_w in listwords):
   ind = imgs[i]
   X_test[i,indseq,0:100] = encoded_images[ind]# COMPLETE WITH YOUR CODE
   ind = listwords.index(current_w)
   X_test[i,indseq,100:202] = embeddings[ind,:] # COMPLETE WITH YOUR CODE

  next_w = words_in_caption[j+1].lower()
  if(next_w in listwords):
   index_pred = indexwords[next_w]# COMPLETE WITH YOUR CODE
   Y_test[i,indseq,index_pred] = True # COMPLETE WITH YOUR CODE
   indseq += 1 # Increment index if target label present in reduced dictionary

outfile = 'Test_data_'+str(nbkeep)
np.savez(outfile, X_test=X_test, Y_test=Y_test) # Saving tensor


filename = 'flickr_8k_train_dataset.txt'
df = pd.read_csv(filename, delimiter='\t')
nb_samples = df.shape[0]
iter = df.iterrows()

bow = {}
nbwords = 0

for i in range(nb_samples):
 x = iter.__next__()
 cap_words = x[1][1].split() # split caption into words
 cap_wordsl = [w.lower() for w in cap_words] # remove capital letters
 nbwords += len(cap_wordsl)
 for w in cap_wordsl:
  if (w in bow):
   bow[w] = bow[w]+1
  else:
   bow[w] = 1

bown = sorted([(value,key) for (key,value) in bow.items()], reverse=True)


nbkeep = 1000 # 100 is needed for fast processing

outfile = 'Caption_Embeddings.p'
[listwords, embeddings] = pickle.load( open( outfile, "rb" ) )

embeddings_new = np.zeros((nbkeep,102))
listwords_new = []

for i in range(nbkeep):
 listwords_new.append(bown[i][1])
 ind = listwords.index(bown[i][1])
 embeddings_new[i,:] = embeddings[ind,:]# COMPLETE WITH YOUR CODE
 embeddings_new[i,:] /= np.linalg.norm(embeddings_new[i,:]) # Normalization


listwords = listwords_new
embeddings = embeddings_new
outfile = "Caption_Embeddings_"+str(nbkeep)+".p"
with open(outfile, "wb" ) as pickle_f:
 pickle.dump( [listwords, embeddings], pickle_f)


freqnc = np.cumsum([float(w[0])/nbwords*100.0 for w in bown])

x_axis = [str(bown[i][1]) for i in range(100)]
plt.figure(dpi=300)
plt.xticks(rotation=90, fontsize=3)
plt.ylabel('Word Frequency')
plt.bar(x_axis, freqnc[0:100])

print("number of kept words="+str(nbkeep)+" - ratio="+str(freqnc[nbkeep-1])+" %")

#Exercice 2

nbTrain = df.shape[0]
iter = df.iterrows()

caps = [] # Set of captions
imgs = [] # Set of images
for i in range(nbTrain):
 x = iter.__next__()
 caps.append(x[1][1])
 imgs.append(x[1][0])

maxLCap = 0

for caption in caps:
   l=0
   words_in_caption =  caption.split()
   for j in range(len(words_in_caption)-1):
    current_w = words_in_caption[j].lower()
    if(current_w in listwords):
     l+=1
   if(l > maxLCap):
       maxLCap = l

print("max caption length ="+str(maxLCap))

indexwords = {} # Useful for tensor filling
for i in range(len(listwords)):
 indexwords[listwords[i]] = i

# Loading images features
encoded_images = pickle.load( open( "encoded_images_PCA.p", "rb" ) )

# Allocating data and labels tensors
tinput = 202
tVocabulary = len(listwords)
nbTrain = 10000;
X_train = np.zeros((nbTrain,maxLCap, tinput))
Y_train = np.zeros((nbTrain,maxLCap, tVocabulary), bool)

for i in range(nbTrain):
 words_in_caption =  caps[i].split()
 indseq=0 # current sequence index (to handle mising words in reduced dictionary)
 for j in range(len(words_in_caption)-1):
  current_w = words_in_caption[j].lower()
  if(current_w in listwords):
   ind = imgs[i]
   X_train[i,indseq,0:100] = encoded_images[ind]# COMPLETE WITH YOUR CODE
   ind = listwords.index(current_w)
   X_train[i,indseq,100:202] = embeddings[ind,:] # COMPLETE WITH YOUR CODE

  next_w = words_in_caption[j+1].lower()
  if(next_w in listwords):
   index_pred = indexwords[next_w]# COMPLETE WITH YOUR CODE
   Y_train[i,indseq,index_pred] = True # COMPLETE WITH YOUR CODE
   indseq += 1 # Increment index if target label present in reduced dictionary

outfile = 'Training_data_'+str(nbkeep)
np.savez(outfile, X_train=X_train, Y_train=Y_train) # Saving tensor

#Exercice 3

def saveModel(model, savename):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(savename+".yaml", "w") as yaml_file:
     yaml_file.write(model_yaml)
    print("Yaml Model ",savename,".yaml saved to disk")
    # serialize weights to HDF5
    model.save_weights(savename+".h5")
    print("Weights ",savename,".h5 saved to disk")

timesteps = 35;
features = 202;
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(timesteps, features)))
model.add(SimpleRNN(100, input_shape=(timesteps, features), return_sequences=True, unroll=True))
model.add(Dense(1000, name='fc5'))
model.add(Activation('softmax'))

#learning_rate = 1.0
#adam = Adam(learning_rate)
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.summary()

batch_size = 10
nb_epoch = 10
# convert class vectors to binary class matrices
model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1)


# LOADING TEST DATA
outfile = 'Test_data_'+str(1000)+'.npz'
npzfile = np.load(outfile)

X_test = npzfile['X_test']
Y_test = npzfile['Y_test']


scores = model.evaluate(X_train, Y_train, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

saveModel(model,"model_10mil")

#Exercice 4
def loadModel(savename):
 with open(savename+".yaml", "r") as yaml_file:
  model = model_from_yaml(yaml_file.read())
 print ("Yaml Model ",savename,".yaml loaded ")
 model.load_weights(savename+".h5")
 print ("Weights ",savename,".h5 loaded ")
 return model

# LOADING MODEL
nameModel = "model_10mil" # COMPLETE with your model name
model = loadModel(nameModel)

optim = Adam()
model.compile(loss="categorical_crossentropy", optimizer=optim,metrics=['accuracy'])


outfile = "Caption_Embeddings_"+str(1000)+".p"
[listwords, embeddings] = pickle.load( open( outfile, "rb" ) )
indexwords = {}
for i in range(len(listwords)):
 indexwords[listwords[i]] = i

####

import matplotlib.image as mpimg
 
ind = np.random.randint(X_test.shape[0])

filename = 'flickr_8k_test_dataset.txt' #  PATH IF NEEDED

df = pd.read_csv(filename, delimiter='\t')
iter = df.iterrows()

for i in range(ind+1):
 x = iter.__next__()

imname = x[1][0]
print("image name="+imname+" caption="+x[1][1])
dirIm = "Flickr8k_Dataset/" # CHANGE WITH YOUR DATASET

img=mpimg.imread(dirIm+imname)
plt.figure(dpi=100)
plt.imshow(img)
plt.axis('off')
plt.show()

pred = model.predict(X_test[ind:ind+1,:,:])

def sampling(preds, temperature=1.0):
 preds = np.asarray(preds).astype('float64')
 predsN = pow(preds,1.0/temperature)
 predsN /= np.sum(predsN)
 probas = np.random.multinomial(1, predsN, 1)
 return np.argmax(probas)

nbGen = 5
temperature=0.1 # Temperature param for peacking soft-max distribution
X_test1 = X_test
for s in range(nbGen):
 wordpreds = "Caption n° "+str(s+1)+": "
 indpred = sampling(pred[0,0,:], temperature)
 wordpred = listwords[indpred]
 wordpreds +=str(wordpred)+ " "
 X_test1[ind:ind+1,1,100:202] = embeddings[indpred,:] # COMPLETE WITH YOUR CODE
 cpt=1
 while(str(wordpred)!='<end>' and cpt<30):
  pred = model.predict(X_test1[ind:ind+1,:,:])
  indpred = sampling(pred[0,cpt,:], temperature)
  wordpred = listwords[indpred]
  wordpreds += str(wordpred)+ " "
  cpt+=1
  X_test1[ind:ind+1,cpt,100:202] = embeddings[indpred,:] # COMPLETE WITH YOUR CODE
 print(wordpreds)


# LOADING TEST DATA
nbkeep = 1000
outfile = "" # REPLACE WITH YOUR DATA PATH
outfile += 'Test_data_'+str(nbkeep)+'.npz'
npzfile = np.load(outfile)

X_test = npzfile['X_test']
Y_test = npzfile['Y_test']

# LOADING MODEL
#nameModel = "model_10mil" #REPLACE WITH YOUR MODEL NAME
model = loadModel(nameModel)

# COMPILING MODEL
optim = Adam()
model.compile(loss="categorical_crossentropy", optimizer=optim,metrics=['accuracy'])
scores_test = model.evaluate(X_test, Y_test, verbose=1)
print("PERFS TEST: %s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100))

# LOADING TEXT EMBEDDINGS
outfile = "Caption_Embeddings_"+str(nbkeep)+".p"
[listwords, embeddings] = pickle.load( open( outfile, "rb" ) )
indexwords = {}
for i in range(len(listwords)):
 indexwords[listwords[i]] = i

# COMPUTING CAPTION PREDICTIONS ON TEST SET
predictions = []
nbTest = X_test.shape[0]

for i in range(0,nbTest,5):
 pred = model.predict(X_test[i:i+1,:,:])
 wordpreds = []
 indpred = np.argmax(pred[0,0,:])
 wordpred = listwords[indpred]
 wordpreds.append(str(wordpred))
 X_test[i,1,100:202] = embeddings[indpred]
 cpt=1
 while(str(wordpred)!='<end>' and cpt<(X_test.shape[1]-1)):
  pred = model.predict(X_test[i:i+1,:,:])
  indpred = np.argmax(pred[0,cpt,:])
  wordpred = listwords[indpred]
  if(wordpred !='<end>'):
   wordpreds.append(str(wordpred))
  cpt+=1
  X_test[i,cpt,100:202] = embeddings[indpred]

 if(i%1000==0):
  print("i="+str(i)+" "+str(wordpreds))
 predictions.append(wordpreds)

# LOADING GROUD TRUTH CAPTIONS ON TEST SET
references = []
filename = 'flickr_8k_test_dataset.txt'
df = pd.read_csv(filename, delimiter='\t')
iter = df.iterrows()

ccpt =0
for i in range(int(nbTest/5)):
 captions_image = []
 for j in range(5):
  x = iter.__next__()
  ll = x[1][1].split()
  caption = []
  for k in range(1,len(ll)-1):
   caption.append(ll[k])

  captions_image.append(caption)
  ccpt+=1

 references.append(captions_image)

# COMPUTING BLUE-1, BLUE-2, BLUE-3, BLUE-4
blue_scores = np.zeros(4)
weights = np.zeros((4,4))
weights[0,0] = 1
weights[1,0] = 0.5
weights[1,1] = 0.5
weights[2,0] = 1.0/3.0
weights[2,1] = 1.0/3.0
weights[2,2] = 1.0/3.0
weights[3,:] = 1.0/4.0

for i in range(4):
 blue_scores[i] = nltk.translate.bleu_score.corpus_bleu(references, predictions, weights = (weights[i,0], weights[i,1], weights[i,2], weights[i,3]) )
 print("blue_score - "+str(i)+"="+str( blue_scores[i]))


















