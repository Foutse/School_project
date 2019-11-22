# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 16:38:10 2019

@author: FOUTSE
"""

#Feature Importance with datasets.load_iris() # fit an ExtraPython

# Feature Importance
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import matplotlib.pyplot as plt

# load the iris datasets
#dataset_iris = datasets.load_iris()
#data = dataset_iris.data
#dataT = dataset_iris.target
#donneeMer = np.load("donneeSurMernoNan.npy")
datas0 = np.load("donnees0.npy")
datas1 = np.load("donnees1.npy")
datas2 = np.load("donnees2.npy")
datas3 = np.load("donnees3.npy")
datas4 = np.load("donnees4.npy")
datas5 = np.load("donnees5.npy")
datas6 = np.load("donnees6.npy")
datas7 = np.load("donnees7.npy")
datas8 = np.load("donnees8.npy")
datas9 = np.load("donnees9.npy")
datas10 = np.load("donnees10.npy")
datas11 = np.load("donnees11.npy")
datas12 = np.load("donnees12.npy")
datas13 = np.load("donnees13.npy")
datas14 = np.load("donnees14.npy")
datas15 = np.load("donnees15.npy")
datas16 = np.load("donnees16.npy")
datas17 = np.load("donnees17.npy")
datas18 = np.load("donnees18.npy")
datas19 = np.load("donnees19.npy")
datas20 = np.load("donnees20.npy")
datas21 = np.load("donnees21.npy")
datas22 = np.load("donnees22.npy")
datas23 = np.load("donnees23.npy")
datas24 = np.load("donnees24.npy")
datas25 = np.load("donnees25.npy")
datas26 = np.load("donnees26.npy")
datas27 = np.load("donnees27.npy")
datas28 = np.load("donnees28.npy")
datas29 = np.load("donnees29.npy")
datas30 = np.load("donnees30.npy")
datas31 = np.load("donnees31.npy")
datas32 = np.load("donnees32.npy")
datas33 = np.load("donnees33.npy")
datas34 = np.load("donnees34.npy")
datas35 = np.load("donnees35.npy")
datas36 = np.load("donnees36.npy")
datas37 = np.load("donnees37.npy")
datas38 = np.load("donnees38.npy")
datas39 = np.load("donnees39.npy")
datas40 = np.load("donnees40.npy")
datas41 = np.load("donnees41.npy")
datas42 = np.load("donnees42.npy")
datas43 = np.load("donnees43.npy")
datas44 = np.load("donnees44.npy")
datas45 = np.load("donnees45.npy")
datas46 = np.load("donnees46.npy")
datas47 = np.load("donnees47.npy")
datas48 = np.load("donnees48.npy")
datas49 = np.load("donnees49.npy")
datas50 = np.load("donnees50.npy")
datas51 = np.load("donnees51.npy")
datas52 = np.load("donnees52.npy")
datas53 = np.load("donnees53.npy")
datas54 = np.load("donnees54.npy")
datas55 = np.load("donnees55.npy")
datas56 = np.load("donnees56.npy")
datas57 = np.load("donnees57.npy")
datas58 = np.load("donnees58.npy")
datas59 = np.load("donnees59.npy")
datas60 = np.load("donnees60.npy")
datas61 = np.load("donnees61.npy")
datas62 = np.load("donnees62.npy")
datas63 = np.load("donnees63.npy")
datas64 = np.load("donnees64.npy")
datas65 = np.load("donnees65.npy")
datas66 = np.load("donnees66.npy")
datas67 = np.load("donnees67.npy")
datas68 = np.load("donnees68.npy")
datas69 = np.load("donnees69.npy")
datas70 = np.load("donnees70.npy")
datas71 = np.load("donnees71.npy")
datas72 = np.load("donnees72.npy")
datas73 = np.load("donnees73.npy")
datas74 = np.load("donnees74.npy")
datas75 = np.load("donnees75.npy")
datas76 = np.load("donnees76.npy")
datas77 = np.load("donnees77.npy")
datas78 = np.load("donnees78.npy")
datas79 = np.load("donnees79.npy")
datas80 = np.load("donnees80.npy")
datas81 = np.load("donnees81.npy")
datas82 = np.load("donnees82.npy")
datas83 = np.load("donnees83.npy")
datas84 = np.load("donnees84.npy")
datas85 = np.load("donnees85.npy")
datas86 = np.load("donnees86.npy")
datas87 = np.load("donnees87.npy")
datas88 = np.load("donnees88.npy")
datas89 = np.load("donnees89.npy")
datas90 = np.load("donnees90.npy")
datas91 = np.load("donnees91.npy")

datatrain= np.concatenate((datas0, datas1, datas2,datas4,datas5,datas6,datas8,datas9,datas10,
                            datas12,datas13,datas14,datas16,datas17,datas18,datas20,datas21,datas22, 
                            datas24, datas25, datas26, datas28,datas29,datas30,datas32,datas33,datas34,
                            datas36,datas37,datas38,datas40,datas41,datas42,datas44,datas45,datas46, 
                            datas47,datas48,datas49,datas51,datas52,datas53,datas55,datas56,
                            datas57,datas59,datas60,datas61,datas63,datas64,datas65,
                            datas67,datas68,datas69,datas71,datas72,datas73,datas75,datas76,
                            datas77,datas79,datas80,datas81,datas83,datas84,datas85,
                            datas87,datas88,datas89,datas91), axis=1)

#test data set
datatest = np.concatenate((datas3,datas7,datas11,datas15, datas19,datas23,datas27,datas31,datas35, datas39,
                           datas43,datas50,datas54,datas58,datas62,datas66,datas70,datas74,datas78,datas82,
                           datas86,datas90,), axis=1)

np.save("DataTrain", datatrain)
np.save("DataTest", datatest)
data_for_train = np.load("DataTrain.npy")
data_for_test = np.load("DataTest.npy")

data_for_train = data_for_train.T #datatrain.T #data[0:20,:,:]
#data2 = np.reshape(data1, (7, len(data1[0])))
#dataset = (np.load("donneeSurMernoNan.npy")).T 
data1 = pd.DataFrame(data_for_train[:,0:2], columns=["SST","CHL-OC5_mean"])
data2 = pd.DataFrame(data_for_train[:,3:], columns = ["NRRS555_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean"])
data_train =pd.concat([data1, data2], axis=1)
data_target = pd.DataFrame(data_for_train[:,2], columns = ["PFT"])

#np.save("DonneesTrainx", data_for_train)
#np.save("DonneesTrainxLabel", data_target)

data_for_test = data_for_test.T #datatest.T #data[0:20,:,:]
#data2 = np.reshape(data1, (7, len(data1[0])))
#dataset = (np.load("donneeSurMernoNan.npy")).T 
data1t = pd.DataFrame(data_for_test[:,0:2], columns=["SST","CHL-OC5_mean"])
data2t = pd.DataFrame(data_for_test[:,3:], columns = ["NRRS555_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean"])
data_test =pd.concat([data1t, data2t], axis=1)
data_target_test = pd.DataFrame(data_for_test[:,2], columns = ["PFT"])
#np.save("DonneesTest", data_test)
#np.save("DonneesTestLabel", data_target_test)
np.save("DataTrainDF", data_train)
np.save("DataTestDF", data_test)


fig = plt.figure()

plt.subplot(3, 3, 1)
plt.plot(data_train["CHL-OC5_mean"])#,data_train["CHL-OC5_mean"] )
plt.title("CHL-OC5_mean_Train")
plt.subplot(3, 3, 2)
plt.plot(data_train["SST"])#,data_train["CHL-OC5_mean"] )
plt.title("SST_Train")
plt.subplot(3, 3, 3)
plt.plot(data_train["NRRS555_mean"])#,data_train["CHL-OC5_mean"] )
plt.title("NRRS555_mean_Train")
plt.subplot(3, 3, 4)
plt.plot(data_train["NRRS490_mean"])#,data_train["CHL-OC5_mean"] )
plt.title("NRRS490_mean_Train")
plt.subplot(3, 3, 5)
plt.plot(data_train["NRRS443_mean"])#,data_train["CHL-OC5_mean"] )
plt.title("NRRS443_mean_Train")
plt.subplot(3, 3, 6)
plt.plot(data_train["NRRS412_mean"])#,data_train["CHL-OC5_mean"] )
plt.title("NRRS412_mean_Train")
plt.show()

fig = plt.figure()

plt.subplot(3, 3, 1)
plt.plot(data_test["CHL-OC5_mean"])#,data_train["CHL-OC5_mean"] )
plt.title("CHL-OC5_mean_Test")
plt.subplot(3, 3, 2)
plt.plot(data_test["SST"])#,data_train["CHL-OC5_mean"] )
plt.title("SST_Train")
plt.subplot(3, 3, 3)
plt.plot(data_test["NRRS555_mean"])#,data_train["CHL-OC5_mean"] )
plt.title("NRRS555_mean_Test")
plt.subplot(3, 3, 4)
plt.plot(data_test["NRRS490_mean"])#,data_train["CHL-OC5_mean"] )
plt.title("NRRS490_mean_Test")
plt.subplot(3, 3, 5)
plt.plot(data_test["NRRS443_mean"])#,data_train["CHL-OC5_mean"] )
plt.title("NRRS443_mean_Test")
plt.subplot(3, 3, 6)
plt.plot(data_test["NRRS412_mean"])#,data_train["CHL-OC5_mean"] )
plt.title("NRRS412_mean_Test")
plt.show()


# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(data_train, data_target)

# display the relative importance of each attribute
print(model.feature_importances_)
feature_names = ["SST","CHL-OC5_mean","NRRS555_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean"]
feature_imp = pd.Series(model.feature_importances_,index=feature_names).sort_values(ascending=False)
feature_imp
#visualisation of the important features
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
plt.figure()
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features with ExtraTreesClassifier")
plt.legend()
plt.show()





#Arbre de decision
from sklearn.datasets import *
from sklearn import tree
import graphviz
from IPython.display import Image  
import pydotplus



clf = tree.DecisionTreeClassifier() # init the tree
clf = clf.fit(data_train, data_target) # train the tree

#measuring decision tree performance
from sklearn.tree import DecisionTreeRegressor

#clf = DecisionTreeRegressor().fit(data_train, data_target)

predicted = clf.predict(data_train)
expected = data_target
acc = metrics.accuracy_score(expected, predicted)*100
#from matplotlib import pyplot as plt
plt.figure(figsize=(4, 3))
plt.scatter(expected, predicted)
plt.plot([1, 7], [1, 7], '--k')
plt.axis('tight')
plt.xlabel('True classe')
plt.ylabel('Predicted classe on trained data')
plt.title("Accuracy: %1.2f %%" %acc)
plt.tight_layout()
#print("Accuracy:",metrics.accuracy_score(expected, predicted))

predicted = clf.predict(data_test)
expected = data_target_test

acc = metrics.accuracy_score(expected, predicted)*100
from matplotlib import pyplot as plt
plt.figure(figsize=(4, 3))
plt.scatter(expected, predicted)
plt.plot([1, 7], [1, 7], '--k')
plt.axis('tight')
plt.xlabel('True classe')
plt.ylabel('Predicted classe on test dataset')
plt.title("Decision Tree Accuracy:%1.2f %%" %acc)
plt.tight_layout()

#Random forest
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification


#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(data_train, data_target)
#clf.fit(X_train,y_train)

y_pred=clf.predict(data_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?

acc = metrics.accuracy_score(data_target_test, y_pred)*100
from matplotlib import pyplot as plt
plt.figure(figsize=(4, 3))
plt.scatter(expected, y_pred)
plt.plot([1, 7], [1, 7], '--k')
plt.axis('tight')
plt.xlabel('True classe')
plt.ylabel('Predicted classe on test dataset')
plt.title("RandomForest Accuracy:%1.2f %%" %acc)
plt.tight_layout()

print("Accuracy:",metrics.accuracy_score(data_target_test, y_pred))

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)


#print(clf.feature_importances_)
#[0.14205973 0.76664038 0.0282433  0.06305659]
feature_names = ["SST","CHL-OC5_mean","NRRS555_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean"]
feature_imp = pd.Series(clf.feature_importances_,index=feature_names).sort_values(ascending=False)
#visualisation of the important features
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.figure()
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features (Random Forest)")
plt.legend()
plt.show()



#Classification without SST
newdata1 = pd.DataFrame(data_for_train[:,1], columns=["CHL-OC5_mean"])
newdata2 = pd.DataFrame(data_for_train[:,3:], columns = ["NRRS555_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean"])
newdata_train =pd.concat([newdata1, newdata2], axis=1)
data_target = pd.DataFrame(data_for_train[:,2], columns = ["PFT"])

#Test data set
#Testdataset = (np.load("donnees20.npy")).T #datasets.load_iris()
#cols = ["NRRS555_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean","SST","CHL-OC5_mean"]
newdata1 = pd.DataFrame(data_for_test[:,1], columns=["CHL-OC5_mean"])
newdata2 = pd.DataFrame(data_for_test[:,3:], columns = ["NRRS555_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean"])
newdata_test =pd.concat([newdata1, newdata2], axis=1)
data_test_target = pd.DataFrame(data_for_test[:,2], columns = ["PFT"])



model = ExtraTreesClassifier()
model.fit(newdata_train, data_target)

# display the relative importance of each attribute
print(model.feature_importances_)
feature_names = ["CHL-OC5_mean","NRRS555_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean"]
feature_imp = pd.Series(model.feature_importances_,index=feature_names).sort_values(ascending=False)
feature_imp
#visualisation of the important features
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features with ExtraTreesClassifier ")
plt.legend()
plt.show()


clf = tree.DecisionTreeClassifier() # init the tree
clf = clf.fit(newdata_train, data_target) # train the tree

#measuring decision tree performance
predicted = clf.predict(newdata_test)
expected = data_test_target

acc = metrics.accuracy_score(expected, predicted)*100
from matplotlib import pyplot as plt
plt.figure(figsize=(4, 3))
plt.scatter(expected, predicted)
plt.plot([1, 7], [1, 7], '--k')
plt.axis('tight')
plt.xlabel('True classe')
plt.ylabel('Predicted classe on test dataset')
plt.title("Decision Tree Accuracy:%1.2f %%" %acc)
plt.tight_layout()



#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(newdata_train, data_target)
#clf.fit(X_train,y_train)

y_pred=clf.predict(newdata_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?

acc = metrics.accuracy_score(data_test_target, y_pred)*100
from matplotlib import pyplot as plt
plt.figure(figsize=(4, 3))
plt.scatter(expected, y_pred)
plt.plot([1, 7], [1, 7], '--k')
plt.axis('tight')
plt.xlabel('True classe')
plt.ylabel('Predicted classe on test dataset')
plt.title("RandomForest Accuracy:%1.2f %%" %acc)
plt.tight_layout()


RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)


print(clf.feature_importances_)
#[0.14205973 0.76664038 0.0282433  0.06305659]
feature_names = ["CHL-OC5_mean","NRRS555_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean"]
feature_imp = pd.Series(clf.feature_importances_,index=feature_names).sort_values(ascending=False)
#visualisation of the important features
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
# Creating a bar plot
plt.figure()
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features (Random Forest)")
plt.legend()
plt.show()


#Classification without NRRS555_mean
newdata1 = pd.DataFrame(data_for_train[:,1], columns=["CHL-OC5_mean"])
newdata2 = pd.DataFrame(data_for_train[:,4:], columns = ["NRRS490_mean","NRRS443_mean","NRRS412_mean"])
newdata_train =pd.concat([newdata1, newdata2], axis=1)
data_target = pd.DataFrame(data_for_train[:,2], columns = ["PFT"])

#Test data set
#Testdataset = (np.load("donnees2.npy")).T 
newdata1 = pd.DataFrame(data_for_test[:,1], columns=["CHL-OC5_mean"])
newdata2 = pd.DataFrame(data_for_test[:,4:], columns = ["NRRS490_mean","NRRS443_mean","NRRS412_mean"])
newdata_test =pd.concat([newdata1, newdata2], axis=1)
data_test_target = pd.DataFrame(data_for_test[:,2], columns = ["PFT"])



model = ExtraTreesClassifier()
model.fit(newdata_train, data_target)

# display the relative importance of each attribute
print(model.feature_importances_)
feature_names = ["CHL-OC5_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean"]
feature_imp = pd.Series(model.feature_importances_,index=feature_names).sort_values(ascending=False)
feature_imp
#visualisation of the important features
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features with ExtraTreesClassifier ")
plt.legend()
plt.show()


clf = tree.DecisionTreeClassifier() # init the tree
clf = clf.fit(newdata_train, data_target) # train the tree
## export the learned decision tree
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=["CHL-OC5_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean"],
                         class_names=["PFT1","PFT2","PFT3","PFT4","PFT5", "PFT6","PFT7"],
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("PhytoplanctonTreeMinus555AndSST") # tree saved to PhytoplanctonTree.pdf

#measuring decision tree performance
predicted = clf.predict(newdata_test)
expected = data_test_target

acc = metrics.accuracy_score(expected, predicted)*100
from matplotlib import pyplot as plt
plt.figure(figsize=(4, 3))
plt.scatter(expected, predicted)
plt.plot([1, 7], [1, 7], '--k')
plt.axis('tight')
plt.xlabel('True classe')
plt.ylabel('Predicted classe on test dataset')
plt.title("Decision Tree Accuracy:%1.2f %%" %acc)
plt.tight_layout()


#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(newdata_train, data_target)
#clf.fit(X_train,y_train)

y_pred=clf.predict(newdata_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?

acc = metrics.accuracy_score(data_test_target, y_pred)*100
from matplotlib import pyplot as plt
plt.figure(figsize=(4, 3))
plt.scatter(expected, y_pred)
plt.plot([1, 7], [1, 7], '--k')
plt.axis('tight')
plt.xlabel('True classe')
plt.ylabel('Predicted classe on test dataset')
plt.title("RandomForest Accuracy:%1.2f %%" %acc)
plt.tight_layout()


RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)


print(clf.feature_importances_)
#[0.14205973 0.76664038 0.0282433  0.06305659]
feature_names = ["CHL-OC5_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean"]
feature_imp = pd.Series(clf.feature_importances_,index=feature_names).sort_values(ascending=False)
#visualisation of the important features
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
# Creating a bar plot
plt.figure()
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features (Random Forest)")
plt.legend()
plt.show()

