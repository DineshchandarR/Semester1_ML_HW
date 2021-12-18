# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 23:27:05 2021

@author: Dineshchandar Ravichandran

Course: CPSC-6430-002-91763

Prof: Dr.Nianyi Li

P5: Binary Classification to Predict the Presence or Absence of Breast Cancer

"""

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

#Loading the data and storing in a Dataframe
data = load_breast_cancer()
dataDF = pd.DataFrame(data.data, columns = data.feature_names)
dataDF['Target'] = data.target
dataDF.head()

print('\nNo. of Benign Malignant data =',len(dataDF[dataDF['Target'] == 0]),
      '\nNo. of Benign data =',len(dataDF[dataDF['Target'] == 1]))

# Visualising the data
sns.pairplot(dataDF.iloc[:,-8:], hue='Target')

#Spliting X and Y values
X = dataDF.iloc[:,:-1]
Y = dataDF.iloc[:,-1]

# Test and Traing Data split using SKLEARN
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

##KNN##

print("\n\nKNN\n\n")
from sklearn.neighbors import KNeighborsClassifier

pipe = Pipeline([
    ('scale',StandardScaler()),
    ('Classifier', KNeighborsClassifier())
])

param_tune= [{'Classifier__n_neighbors': np.arange(1,50),},]

knnGSCV = GridSearchCV(pipe,param_tune,cv=5)
knnGSCV.fit(X,Y)
print(knnGSCV.best_params_)
#configuring the n_neighbors based on knnGSCV.best_params_
pipe =  make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=7))
pipe.fit(X_train,Y_train)
KNNPred=pipe.predict(X_test)

PredDf=X_test.copy(deep=True)
PredDf['GTruth']=Y_test
PredDf['KNNPred']=KNNPred
#PredDf.head()

#Confusion Matrix
TN, FP, FN, TP = confusion_matrix(PredDf['GTruth'],PredDf['KNNPred']).ravel()
print(classification_report(PredDf['GTruth'], PredDf['KNNPred']))

#Confusion Matrix resutls method
def CMRes(TN, FP, FN, TP):
    ConfMat=pd.DataFrame(data={' ':['Malignant ','Benign '],
                           'Malignant':['True Malignat: '+str(TN),'Fasle Benign: '+str(FP)],
                           'Benign ':['False Malignat: '+str(FN),'True Benign: '+str(TP)],
                           })
    print('Confusion Matrix Values\n',ConfMat,'\n')
    Acu = (TP+TN)/(TP+TN+FP+FN)
    Pre = TP/(TP+FP)
    Rec = TP/(TP+FN)
    F1 = 2*(1/((1/Pre)+(1/Rec)))
    Acu_dis=round((Acu*100), 2)
    Pre_dis=round((Pre*100), 2)
    Rec_dis=round((Rec*100), 2)
    F1_dis =round((F1*100), 2)
    
    print("\n Accuracy:",Acu_dis,"%")
    print("\n Precision:",Pre_dis,"%")
    print("\n Recall:",Rec_dis,"%")
    print("\n F1:",F1_dis,"%")
    return  Acu_dis, Pre_dis, Rec_dis,F1_dis 

print("\nResults of KNN:\n")
print(CMRes(TN, FP, FN, TP))

#Comapring the values with CV scores
cv_scores = cross_val_score(pipe,X,Y,cv=5)
print(cv_scores)
print('CV score mean for KNN:{}'.format(np.mean(cv_scores)))

##Logistic Regression##
print("\n\nLogistic Regression\n\n")
from sklearn.linear_model import LogisticRegression
pipe = Pipeline([
    ('scale',StandardScaler()),
    ('Classifier', LogisticRegression())
])

param_tune = [
    {
        'Classifier__max_iter': [1000,10000],
        'Classifier__C':[1,10,100,1000],
    },
    ]
LRGSCV = GridSearchCV(pipe,param_tune,cv=5)
LRGSCV.fit(X,Y)
print(LRGSCV.best_params_)

pipe =  make_pipeline(StandardScaler(),LogisticRegression(C=1, max_iter = 1000))
pipe.fit(X_train,Y_train)

LRPred=pipe.predict(X_test)
PredDf['LRPred']=LRPred
#PredDf.head()
TN, FP, FN, TP = confusion_matrix(PredDf['GTruth'],PredDf['LRPred']).ravel()
print(classification_report(PredDf['GTruth'], PredDf['LRPred']))
print("\nResults of LR:\n")
print(CMRes(TN, FP, FN, TP))
cv_scores = cross_val_score(pipe,X,Y,cv=5)
print(cv_scores)
print('CV score mean for LR:{}'.format(np.mean(cv_scores)))

##SVM##
print("\n\nSVM\n\n")
from sklearn.svm import SVC
SVMClasifier = make_pipeline(StandardScaler(), SVC(gamma='auto'))
SVMClasifier.fit(X_train , Y_train)
pipe = Pipeline([
    ('scale',StandardScaler()),
    ('Classifier', SVC())
])

param_tune = [
    {
        'Classifier__kernel': ['rbf','linear'],
        'Classifier__C':[1,10,15,100,1000],
        #'Classifier__max_iter':[1500,1000],
    },
    ]
SVMGSCV = GridSearchCV(pipe,param_tune,cv=5)
SVMGSCV.fit(X,Y)
print(SVMGSCV.best_params_)

pipe =  make_pipeline(StandardScaler(),SVC(kernel='rbf', C=10, max_iter= 1500))
pipe.fit(X_train,Y_train)
SVMPred=pipe.predict(X_test)
PredDf['SVMPred']=SVMPred
#PredDf.head()
TN, FP, FN, TP = confusion_matrix(PredDf['GTruth'],PredDf['SVMPred']).ravel()
print(classification_report(PredDf['GTruth'], PredDf['SVMPred']))
print("\nResults of SVM:\n")
print(CMRes(TN, FP, FN, TP))
cv_scores = cross_val_score(pipe,X,Y,cv=5)
print(cv_scores)
print('CV score mean for SVM:{}'.format(np.mean(cv_scores)))

##Multilayer Perceptron##
print("\n\nMultilayer Perceptron\n\n")
from sklearn.neural_network import MLPClassifier

pipe = Pipeline([
    ('scale',StandardScaler()),
    ('Classifier', MLPClassifier())
])

param_tune = [
    {
        'Classifier__hidden_layer_sizes': [(100,),(10,30,10),(20,)],
        'Classifier__activation':['tanh','relu'],
        'Classifier__solver':['sgd','adam'],
        'Classifier__alpha':[0.01,0.001,0.1],
        'Classifier__max_iter':[1000],
        'Classifier__learning_rate':['constant','adaptive'],
    },
    ]

MLPGSCV = GridSearchCV(pipe,param_tune,cv=5)
MLPGSCV.fit(X,Y)
print(MLPGSCV.best_params_)
pipe =  make_pipeline(StandardScaler(),MLPClassifier(
    random_state=0, max_iter=1000, activation='tanh',alpha=0.01,learning_rate='constant',solver='sgd'))
pipe.fit(X_train,Y_train)
MLPPred=pipe.predict(X_test)
PredDf['MLPPred']=MLPPred
#PredDf.head()
TN, FP, FN, TP = confusion_matrix(PredDf['GTruth'],PredDf['MLPPred']).ravel()
print(classification_report(PredDf['GTruth'], PredDf['MLPPred']))
print("\nResults of MLP:\n")
print(CMRes(TN, FP, FN, TP))
cv_scores = cross_val_score(pipe,X,Y,cv=5)
print(cv_scores)
print('CV score mean for MLP:{}'.format(np.mean(cv_scores)))

