# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 14:36:32 2019

@author: Abhranil
"""
##Necessary packages imported
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 

##Input
data=pd.read_csv("train.csv")
y=data["label"].values
x=data.drop("label",axis=1).values

##Splitting the dataset (80% Training, 20% Tesing)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

####SVM Classifier
# training a linear SVM classifier 
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 
  
# model accuracy for X_test   
accuracySVM = svm_model_linear.score(X_test, y_test)
print("SVM"+str(accuracySVM))

####Gaussian Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train, y_train) 
gnb_predictions = gnb.predict(X_test) 
  
# accuracy on X_test 
accuracyGNB = gnb.score(X_test, y_test) 
print("GaussianNB"+str(accuracyGNB)) 
  

##KNN classifier
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 
  
# accuracy on X_test 
accuracy = knn.score(X_test, y_test)
knn_predictions = knn.predict(X_test)  
cm = confusion_matrix(y_test, knn_predictions) 
print (accuracy)


##Saving in csv file
test=pd.read_csv("test.csv").values
predictions=knn.predict(test)


out=pd.DataFrame(predictions)
out.index = np.arange(1, len(out)+1)
out.columns=["Label"]
out.to_csv("prediction.csv",index_label="ImageId")
#predictions.to_csv("prediction.csv")