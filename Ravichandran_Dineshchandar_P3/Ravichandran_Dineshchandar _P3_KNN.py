# -*- coding: utf-8 -*-
"""
Created on Fri Oct 1 15:36:06 2021

@author: Dineshchandar Ravichandran

Course: CPSC-6430-002-91763

Prof: Dr.Nianyi Li

P3_MidTerm: Classification with k-Nearest Neighbor

"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

#Calculate Euclidean Distance
def euclideanDist(x1, x2,r):
    sumDist=0
    for i in range (r):
        #print(x1[i] , x2[i])
        sumDist+=(x1[i] - x2[i]) ** 2
    return np.sqrt(sumDist)

#To rearrange the class count based on the highest no.of occurance
def Sort_Tuple(tup):  
    #to sort using second element of tuple
    tup.sort(key = lambda x: x[1]) 
    return tup

#KNN Method
def knn(test,train,k,r):
    # Compute distances between x and all examples in the training set
    distArr=[]
    for i in range (len(train)):
        #print(train[i])
        ed= euclideanDist(test,train[i],r)
        distArr.append((train[i],ed))
    
    # Sort by distance and return indices of the first k neighbors
    distArr_Sorted=Sort_Tuple(distArr)
    #print(distArr_Sorted)
    knn=[]
    for i in range (k):
        knn.append(distArr_Sorted[i])
    return knn

#Predict based on KNN Value
def predict(KNN):
    votes= {}
    for i in range (len (KNN)):
        pred= KNN[i][0][len(KNN[0][0])-1]
        if pred in votes:
            votes[pred]+= 1
        else:
            votes[pred]= 1
    #print(votes)
    finalVote= {k: v for k, v in sorted(votes.items(), key=lambda item: item[1],reverse=True)}
    #print(finalVote)
    finalVoteKey = list(finalVote)
    return finalVoteKey[0]

#File reading method
def fileUpload():
    global fname
    fname= input ("Enter the Input File path:\n{eg:G:\File.txt}\n:")
    fpathValid="\.(txt)"
    #File path validation
    if (re.search(fpathValid,fname)):
        return fname
    else:
        print("wrong file format")

try:    
    def loadData():
        fileUpload()
#         return pd.read_csv(fname,sep="\t",skiprows=1)
        return pd.read_csv(fname,sep="\t",header=None,names=["X1","X2","QC"],skiprows=0)

except FileNotFoundError:
    j=0
    if j<=3:
        print("File not found kindly, check the file you have uploaded")
        j+=1
        fileUpload()
    else:
        print("Sorry Reached Max Attempts!!")

print("Enter Training Data Details:\n")
Train=loadData()
Train=Train[1:]
print("\n\nEnter Test Data Details:\n")
Test=loadData()
Test=Test[1:]

# Executing KNN for entrie Test Data and predicting the values:
def KNN_EXE(test,train,k,r):
    KNN_values={}
    KNN_pred={}
    for i in range (len(test)):
        test_val=test[i]
        knnRes=knn(test_val,train,k,r)
        #KNN_values[test_val]=knnRes  // Cannot assign a array as an index
        KNN_values[i]=knnRes
        pred=predict(knnRes)
        KNN_pred[i]=pred
    return KNN_pred

#K-Fold Valuation for finding best K value

#genrating k values(odd 1-21)
k_val=list(range(1, 22, 2))
# KFold obj creation
KF= KFold(n_splits=5,shuffle=False)
#Creation of Test and Train Data as Array
TrainArr =  Train.to_numpy()
TestArr = Test.to_numpy()
# KFold DF to store data summary
k_col_arr=[]
for i in k_val:
    k_str=str(i)
    k_col= "K Value:"+k_str
    k_col_arr.append(k_col)
KNN_val_kfold=pd.DataFrame(columns=k_col_arr)
# KFold K Value Evaluation
foldNo=0
DF_row=[]
folNoArr=[]
KNN_KF_PRED={}
for kftrain, kftest in KF.split(TrainArr):
    KFtrain_arr=[]
    KFtest_arr=[]
    KNN_KF_MC=[]
    foldNo +=1
    folNoArr.append(foldNo)
    for i in range (len(kftrain)):
        KF_train=TrainArr[kftrain[i]]
        KFtrain_arr.append(KF_train)
    for i in range (len(kftest)):
        KF_test=TrainArr[kftest[i]]
        KFtest_arr.append(KF_test)
    for k in (k_val):
        KNN_PRED = KNN_EXE(KFtest_arr,KFtrain_arr,k,2)
        KNN_KF_PRED[k] = KNN_PRED
        MC=0
        for i in range(len(KNN_PRED)):
            if KNN_PRED[i] != KFtest_arr[i][len(KFtest_arr[i])-1]:
                MC += 1
        KNN_KF_MC.append(MC)
    DF_row.append(KNN_KF_MC) 
folNoArr.append('Sum')

#Creating DF to visualise the data
for i in range(len(DF_row)):
    KNN_val_kfold.loc[i]=DF_row[i]
print(KNN_val_kfold)

sumArr=[]
sumArr=KNN_val_kfold.sum()
KNN_summary=KNN_val_kfold
KNN_summary.loc[len(KNN_summary)]=sumArr
KNN_summary.insert(loc=0, column='k_Fold_No.', value=folNoArr)
print(KNN_summary)

#KValue Vs Error Graph

y_axis=list(range(1, 11, 1))
plt.plot(k_val,sumArr,c='Blue',marker="o")
plt.title("K_Fold validation")
plt.ylabel('Error_Sum', fontsize=14)
plt.xlabel('K_Value', fontsize=14)
KVE=plt.show()
print("KValue Vs Error Graph \n", KVE)

# K Accuracy Plot
Acc_k_arr=[]
for i in sumArr:
    Acc_k_p = (1-(i/(85)))*100
    Acc_k_arr.append(Acc_k_p)
plt.plot(k_val,Acc_k_arr,c='red',marker="s")
plt.title("K_Accuracy")
plt.ylabel('Cross Validation Accuracy', fontsize=14)
plt.xlabel('Value of knn', fontsize=14)
ACCPLT=plt.show()
print("K Accuracy Plot \n", ACCPLT)

#Executing KNN on Test and Train datasets using the best K=5 as per above k-fold validation
KNN_PRED_ACT=KNN_EXE(TestArr,TrainArr,5,2)
KNN_PRED_ACT_ARR=[]
for i in range (len(KNN_PRED_ACT)):
    KNN_PRED_ACT_ARR.append(KNN_PRED_ACT[i])
Test['Predicted_data']=KNN_PRED_ACT_ARR

TP=len(Test[(Test['Predicted_data']== 1) & (Test['QC'] == 1)])
TN=len(Test[(Test['Predicted_data']==0) & (Test['QC']==0)])
FP=len(Test[(Test['Predicted_data']==1) & (Test['QC']==0)])
FN=len(Test[(Test['Predicted_data']==0) & (Test['QC']==1)])

#Confusion Matrix Evaluation
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

#Matrix Illusration

ConfMat=pd.DataFrame(data={' ':['0','1'],
                           '0':['True Negatives\n'+str(TN),'Fasle Positives\n'+str(FP)],
                           '1':['False Negatives\n'+str(FN),'True Positives\n'+str(TP)],
                           })

print(ConfMat)

#Table for Consfusion Matrix
height = 9
width = 9

fig, ax  = plt.subplots()
table = ax.table(cellText=ConfMat.values,colLabels = ConfMat.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(15)
cell_height = 1 / (len(ConfMat.values))
for pos, cell in table.get_celld().items():
    cell.set_height(cell_height)
print("Confusion Matrix")
ax.axis("off")
CMAT=plt.show()  

print("Table for Consfusion Matrix \n", CMAT)
