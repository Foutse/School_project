# Load ResNet50 architecture & its weights
from keras.applications.resnet50 import ResNet50
import numpy as np
from sklearn.metrics import average_precision_score
from data_gen import PascalVOCDataGenerator
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

model = ResNet50(include_top=True, weights='imagenet')
model.layers.pop()
# Modify top layers
#I added
data_dir = 'VOCdevkit/VOC2007/' # A changer avec votre chemin
data_generator_train = PascalVOCDataGenerator('trainval', data_dir)
#end I added

x = model.layers[-1].output
x = Dense(data_generator_train.nb_classes, activation='sigmoid', name='predictions')(x)
model = Model(input=model.input,output=x)

count=0;
for layer in model.layers:
    count +=1;
print(count)

nlayers = count ;

for i in range(nlayers):
  model.layers[i].trainable = True

lr = 0.1
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=lr), metrics=['binary_accuracy'])

batch_size=32
nb_epochs=10
data_generator_train = PascalVOCDataGenerator('trainval', data_dir)
steps_per_epoch_train = int(len(data_generator_train.id_to_label) / batch_size) + 1

 

default_batch_size = 200
default_data_dir = 'VOCdevkit/VOC2007/'
subset='test'

def evaluate(model, subset, batch_size=default_batch_size, data_dir=default_data_dir, verbose=0):
    """evaluate
    Compute the mean Average Precision metrics on a subset with a given model

    :param model: the model to evaluate
    :param subset: the data subset
    :param batch_size: the batch which will be use in the data generator
    :param data_dir: the directory where the data is stored
    :param verbose: display a progress bar or not, default is no (0)
    """
    #disable_tqdm = (verbose == 0)

    # Create the generator on the given subset
    data_generator = PascalVOCDataGenerator(subset, data_dir)
    steps_per_epoch = int(len(data_generator.id_to_label) / batch_size) + 1

    # Get the generator
    generator = data_generator.flow(batch_size=batch_size)

    y_all = []
    y_pred_all = []
    for i in range(steps_per_epoch):
        # Get the next batch
        X, y = next(generator)
        y_pred = model.predict(X)
        # We concatenate all the y and the prediction
        for y_sample, y_pred_sample in zip(y, y_pred):
            y_all.append(y_sample)
            y_pred_all.append(y_pred_sample)
    y_all = np.array(y_all)
    y_pred_all = np.array(y_pred_all)

    # Now we can compute the AP for each class
    AP = np.zeros(data_generator.nb_classes)
    for cl in range(data_generator.nb_classes):
        AP[cl] = average_precision_score(y_all[:, cl], y_pred_all[:, cl])

    return AP

prct = evaluate(model, subset, batch_size=default_batch_size, data_dir=default_data_dir, verbose=1)