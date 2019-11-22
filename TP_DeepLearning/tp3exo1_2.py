from keras.applications.resnet50 import ResNet50
from data_gen import PascalVOCDataGenerator
from keras.models import Model
from keras.optimizers import SGD
import numpy as np

model = ResNet50(include_top=True, weights='imagenet')
model.layers.pop()
model = Model(input=model.input,output=model.layers[-1].output)
model.summary()
lr = 0.1
model.compile(loss='binary_crossentropy', optimizer=SGD(lr,momentum=0.9), metrics=['binary_accuracy'])

data_dir = 'VOCdevkit/VOC2007/' # A changer avec votre chemin
data_generator_train = PascalVOCDataGenerator('trainval', data_dir)
#generator = data_generator_train.flow(batch_size=batch_size)

batch_size = 32
generator = data_generator_train.flow(batch_size=batch_size)
# Initilisation des matrices contenant les Deep Features et les labels
X_train = np.zeros((len(data_generator_train.images_ids_in_subset),2048))
Y_train = np.zeros((len(data_generator_train.images_ids_in_subset),20))

# Calcul du nombre e batchs
nb_batches = int(len(data_generator_train.images_ids_in_subset) / batch_size) + 1

for i in range(nb_batches):
    # Pour chaque batch, on extrait les images d'entrée X et les labels y
    X, y = next(generator)
    # On récupère les Deep Feature par appel à predict
    y_pred = model.predict(X)
    X_train[i*batch_size:(i+1)*batch_size,:] = y_pred
    Y_train[i*batch_size:(i+1)*batch_size,:] = y

data_generator_test = PascalVOCDataGenerator('test', data_dir)
generator = data_generator_test.flow(batch_size=batch_size)
X_test = np.zeros((len(data_generator_test.images_ids_in_subset),2048))
Y_test = np.zeros((len(data_generator_test.images_ids_in_subset),20))
# Calcul du nombre e batchs
nb_batches = int(len(data_generator_test.images_ids_in_subset) / batch_size) + 1
for i in range(nb_batches):
    # Pour chaque batch, on extrait les images d'entrée X et les labels y
    X, y = next(generator)
    # On récupère les Deep Feature par appel à predict
    y_pred = model.predict(X)
    X_test[i*batch_size:(i+1)*batch_size,:] = y_pred
    Y_test[i*batch_size:(i+1)*batch_size,:] = y


outfile = 'DF_ResNet50_VOC2007'
np.savez(outfile, X_train=X_train, Y_train=Y_train,X_test=X_test, Y_test=Y_test)
