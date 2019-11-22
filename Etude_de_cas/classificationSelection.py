# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 16:38:10 2019

"""

#Feature Importance with datasets.load_iris() # fit an ExtraPython

# Feature Importance
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd

# load the iris datasets
#dataset_iris = datasets.load_iris()
#data = dataset_iris.data
#dataT = dataset_iris.target
#donneeMer = np.load("donneeSurMernoNan.npy")
dataset = (np.load("donnees1.npy")).T 
data1 = pd.DataFrame(dataset[:,0:2], columns=["SST","CHL-OC5_mean"])
data2 = pd.DataFrame(dataset[:,3:], columns = ["NRRS555_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean"])
data_train =pd.concat([data1, data2], axis=1)
data_target = pd.DataFrame(dataset[:,2], columns = ["PFT"])

#Test data set
Testdataset = (np.load("donnees20.npy")).T #datasets.load_iris()
#cols = ["NRRS555_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean","SST","CHL-OC5_mean"]
data1 = pd.DataFrame(Testdataset[:,0:2], columns=["SST","CHL-OC5_mean"])
data2 = pd.DataFrame(Testdataset[:,3:], columns = ["NRRS555_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean"])
data_test =pd.concat([data1, data2], axis=1)
data_test_target = pd.DataFrame(Testdataset[:,2], columns = ["PFT"])

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
plt.title("Visualizing Important Features")
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
## export the learned decision tree
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=["SST","CHL-OC5_mean","NRRS555_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean"],
                         class_names=["PFT1","PFT2","PFT3","PFT4","PFT5", "PFT6","PFT7"],
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("PhytoplanctonTree") # tree saved to PhytoplanctonTree.pdf

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
#

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

print("Accuracy:",metrics.accuracy_score(data_test_target, y_pred))


RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)


print(clf.feature_importances_)
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
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features (Random Forest)")
plt.legend()
plt.show()



#Classification without SST
newdata1 = pd.DataFrame(dataset[:,1], columns=["CHL-OC5_mean"])
newdata2 = pd.DataFrame(dataset[:,3:], columns = ["NRRS555_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean"])
newdata_train =pd.concat([newdata1, newdata2], axis=1)
data_target = pd.DataFrame(dataset[:,2], columns = ["PFT"])

#Test data set
Testdataset = (np.load("donnees20.npy")).T #datasets.load_iris()
#cols = ["NRRS555_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean","SST","CHL-OC5_mean"]
newdata1 = pd.DataFrame(Testdataset[:,1], columns=["CHL-OC5_mean"])
newdata2 = pd.DataFrame(Testdataset[:,3:], columns = ["NRRS555_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean"])
newdata_test =pd.concat([newdata1, newdata2], axis=1)
data_test_target = pd.DataFrame(Testdataset[:,2], columns = ["PFT"])



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
## export the learned decision tree
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=["SST","CHL-OC5_mean","NRRS555_mean","NRRS490_mean","NRRS443_mean","NRRS412_mean"],
                         class_names=["PFT1","PFT2","PFT3","PFT4","PFT5", "PFT6","PFT7"],
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("PhytoplanctonTree") # tree saved to PhytoplanctonTree.pdf

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
newdata1 = pd.DataFrame(dataset[:,1], columns=["CHL-OC5_mean"])
newdata2 = pd.DataFrame(dataset[:,4:], columns = ["NRRS490_mean","NRRS443_mean","NRRS412_mean"])
newdata_train =pd.concat([newdata1, newdata2], axis=1)
data_target = pd.DataFrame(dataset[:,2], columns = ["PFT"])

#Test data set
Testdataset = (np.load("donnees2.npy")).T 
newdata1 = pd.DataFrame(Testdataset[:,1], columns=["CHL-OC5_mean"])
newdata2 = pd.DataFrame(Testdataset[:,4:], columns = ["NRRS490_mean","NRRS443_mean","NRRS412_mean"])
newdata_test =pd.concat([newdata1, newdata2], axis=1)
data_test_target = pd.DataFrame(Testdataset[:,2], columns = ["PFT"])



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

