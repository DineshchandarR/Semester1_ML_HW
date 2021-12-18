# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 17:54:17 2021

@author: Dineshchandar Ravichandran,C19657741

Course: CPSC-6430-002-91763

Prof: Dr.Nianyi Li

Project: Logistic Regression (MidTerm)
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

##File Upload block start##
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
##File Upload block end##

##Logistic Regression Block Start ##

# Sigmoid Function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Logistic Regression
def LRegression (X,Y,a,itr):
    #X,Y,a=alpha, itr=iteration
    costArr=[]
    n,m=X.shape
    print(n,m)
    w = np.zeros((n,1)) #weight intiation

    
    for i in range(itr):
        linerReg=np.dot(w.T,X)
        #linerReg=np.dot(w,X.T)
        y_pred=sigmoid(linerReg)
        #print('@i=',i,'linerReg=',linerReg,'\nPred=',y_pred,'\nYPredShape=',y_pred.shape)
    
        #Cost Function  
        cost= -(1/m)*np.sum(Y*np.log(y_pred)+ (1-Y)*np.log(1-y_pred))
        #print(cost)
        
        #Gradient Descent
        dW = (1/m)*np.dot(y_pred-Y,X.T)
        #print('\ndw:',dW)
        
        #update weights
        w = w-a*dW.T
        #print('\nw:',w)
        
        #updating costArr
        costArr.append(cost)
        
        if (i%(itr/10) == 0):
            print("Cost at iteration:",i,"is=",cost)
            
    return w,costArr,linerReg,y_pred

#Prediction Function
def predictCls(pred):
    m,n=pred.shape
    y_predicted_cls=[]
    for i in range (n):
        if pred[0][i] > 0.5:
            cls=1
        else:
            cls=0
        y_predicted_cls.append(cls)
    return y_predicted_cls

#Prediction for Test_data
def PredTest(X,w):
    linerReg=np.dot(w.T,X)
    y_pred=sigmoid(linerReg)
    predTestCls=predictCls(y_pred)
    return predTestCls

##Logistic Regression Block End ##


# J Vs Itration 
def JVSIttr(itr,costArr):
    plt.plot(np.arange(itr),costArr,c='blue',marker="v")
    plt.title("Data Visualization")
    plt.ylabel('Cost', fontsize=14)
    plt.xlabel('Number_of_Iterations_of_Gradient_Descent', fontsize=14)
    JVSIttrShow= plt.show()
    return JVSIttrShow

#TPTNFPFN method
def TrueFalseValue(DataFrame,comp1,comp2):
    TP=len(DataFrame[(DataFrame[comp1]== 1) & (DataFrame[comp2] == 1)])
    TN=len(DataFrame[(DataFrame[comp1]==0) & (DataFrame[comp2]==0)])
    FP=len(DataFrame[(DataFrame[comp1]==1) & (DataFrame[comp2]==0)])
    FN=len(DataFrame[(DataFrame[comp1]==0) & (DataFrame[comp2]==1)])
    return TP,TN,FP,FN

#Confusion Matrix Evaluation
def confMatEval(TP,TN,FP,FN):
    Acu = (TP+TN)/(TP+TN+FP+FN)
    Pre = TP/(TP+FP)
    Rec = TP/(TP+FN)
    F1 = 2*(1/((1/Pre)+(1/Rec)))        
    Acu_dis =round((Acu*100), 2)
    Pre_dis =round((Pre*100), 2)
    Rec_dis =round((Rec*100), 2)
    F1_dis  =round((F1*100), 2)
    
    print("\n Accuracy:",Acu_dis,"\n Precision:",Pre_dis,"\n Recall:",Rec_dis,"\n F1:",F1_dis)
    ConfMat=pd.DataFrame(data={' ':['0','1'],
                               '0':['True Negatives\\n'+str(TN),'Fasle Positives\\n'+str(FP)],
                               '1':['False Negatives\\n'+str(FN),'True Positives\\n'+str(TP)],
                               })
    #Matrix Illusration
    #height = 9
    #width = 9
    fig, ax  = plt.subplots()
    table = ax.table(cellText=ConfMat.values,colLabels = ConfMat.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    cell_height = 1 / (len(ConfMat.values))
    for pos, cell in table.get_celld().items():
        cell.set_height(cell_height)
    print("Confusion Matrix")
    ax.axis("off")
    table=plt.show()
    return ConfMat,table

'''
##Training for sample data start##
        
# Sample data set to test the Logistic Module
smapleFile="F:/Clemson/COURSE/SEM-1/Machine Learning Implementation and Evaluation(CPSC6430)/Project1/Irisdata.txt"
sample=pd.read_csv(smapleFile,sep="\t",header=None,names=['sepal_length','sepal_width','petal_length','petal_width','species'],skiprows=1 )

#modifiing data for clasiffication application
for i in range(len(sample)) :
    if sample.loc[i]['species'] == 'Setosa':
        sample.loc[i,['species']] = 1
    else:
        sample.loc[i,['species']] = 0
        
        
Sample_train,Sample_test = train_test_split(sample, test_size=0.40, random_state=42)
print(Sample_train,'\n',Sample_test)

x_train = (Sample_train.drop("species",axis=1)).values
y_train = (Sample_train.species).values
x_test  = (Sample_test.drop("species",axis=1)).values
y_test  = (Sample_test.species).values
x_train = x_train.T
y_train = y_train.reshape(1,x_train.shape[1])
x_test  = x_test.T
y_test  = y_test.reshape(1,x_test.shape[1])
print("Shape of x_train=",x_train.shape,"\nShape of y_train=",y_train.shape,"\nShape of x_test",x_test.shape,"\nShape of y_test=",y_test.shape)

#Data TypeDefination for X & Y
x_train = np.array(x_train)
y_train = np.array(y_train,dtype=np.float64)

print(x_train.dtype,y_train.dtype)

SampleItr= 10000
SampleA=0.01
W_Strain, costArr_Strain,linerReg_Strain,y_pred_Strain=LRegression(x_train,y_train,SampleA,SampleItr)

# J Vs Itration Visulization for Sample Train
JVSIttr(SampleItr,costArr_Strain)

#Prediction for Species using the W for lowest J
STrain_yPred_cls=predictCls(y_pred_Strain)
print(STrain_yPred_cls)

#Sample Train Prediction Evaluation Block
sampleT_Disp=Sample_train
sampleT_Disp['Predicted_data']=STrain_yPred_cls
sampleT_Disp

TP_STrain,TN_STrain,FP_STrain,FN_STrain=TrueFalseValue(sampleT_Disp,'Predicted_data','species')
ConfMatSTrain,tableSTrain=confMatEval(TP_STrain,TN_STrain,FP_STrain,FN_STrain)
print(ConfMatSTrain,'\n',tableSTrain)
##Training for sample data end##

##Testing on sample data##
x_test = np.array(x_test)
y_test = np.array(y_test,dtype=np.float64)
print(x_test.dtype,y_test.dtype)

testPred=PredTest(x_test,W_Strain)

sampleTest_Disp=Sample_test
sampleTest_Disp['Predicted_data']=testPred

TP_STest,TN_STest,FP_STest,FN_STest=TrueFalseValue(sampleTest_Disp,'Predicted_data','species')
ConfMatSTest,tableSTest=confMatEval(TP_STest,TN_STest,FP_STest,FN_STest)
print(ConfMatSTest,'\n',tableSTest)
##Testing on sample end##
'''

#Defining Test and Train Data set
print("\nEnter Training Data Details:\n")
Train=loadData()
Train=Train[1:]
print("\n\nEnter Test Data Details:\n")
Test=loadData()
Test=Test[1:]

# Visualising the data
yaxis=list(range(1, 100, 1))
plt.scatter(Train['X1'][Train['QC']==1],Train['X2'][Train['QC']==1],c='green',marker="o")
plt.scatter(Train['X1'][Train['QC']==0],Train['X2'][Train['QC']==0],c='red',marker="x")
plt.scatter(Test['X1'][Test['QC']==1],Test['X2'][Test['QC']==1],c='blue',marker="o")
plt.scatter(Test['X1'][Test['QC']==0],Test['X2'][Test['QC']==0],c='black',marker="x")
plt.title("Data Visualization")
VDACT=plt.show()
print(VDACT)

#Adding Features to exisitng data
def AddFeatures(Data):
    Data_local=Data.copy(deep=True)
    Data_local.insert(loc=2, column='X3', value=(Data_local.X1)**2)
    Data_local.insert(loc=3, column='X4', value=(Data_local.X2)**2)
    Data_local.insert(loc=4, column='X5', value=(Data_local.X3)**2)
    Data_local.insert(loc=5, column='X6', value=(Data_local.X4)**2)
    Data_local.insert(loc=4, column='X7', value=(Data_local.X1)*(Data_local.X2))
    Data_local.insert(loc=5, column='X8', value=(Data_local.X7)**2)
    return Data_local

ActLocalTrain = AddFeatures(Train)
ActLocalTest = AddFeatures(Test)


#Data Assignment for processing Test and Train for actual data
Act_train=ActLocalTrain
Act_test=ActLocalTest
x_ActTrain = (Act_train.drop("QC",axis=1)).values
y_ActTrain = (Act_train.QC).values
x_ActTest  = (Act_test.drop("QC",axis=1)).values
y_ActTest  = (Act_test.QC).values
x_ActTrain = x_ActTrain.T
y_ActTrain = y_ActTrain.reshape(1,x_ActTrain.shape[1])
x_ActTest  = x_ActTest.T
y_ActTest  = y_ActTest.reshape(1,x_ActTest.shape[1])
print("Shape of x_train=",x_ActTrain.shape,"\nShape of y_train=",y_ActTrain.shape,"\nShape of x_test",x_ActTest.shape,"\nShape of y_test=",y_ActTest.shape)


#Data TypeDefination for X & Y
x_ActTrain = np.array(x_ActTrain)
y_ActTrain = np.array(y_ActTrain,dtype=np.float64)
x_ActTest = np.array(x_ActTest)
y_ActTest = np.array(y_ActTest,dtype=np.float64)

print(x_ActTrain.dtype,y_ActTrain.dtype,x_ActTest.dtype,y_ActTest.dtype)

#Actual Training

ActItr= 50000
ActA=1
W_Atrain, costArr_Atrain,linerReg_Atrain,y_pred_Atrain=LRegression(x_ActTrain,y_ActTrain,ActA,ActItr)

# J Vs Itration Visulization for Actual Train Data
JVSIttr(ActItr,costArr_Atrain)

#Prediction for QC using the W for lowest J
ATrain_yPred_cls=predictCls(y_pred_Atrain)
print(ATrain_yPred_cls)

#Actual Train Prediction Evaluation Block
ActT_Disp=ActLocalTrain
ActT_Disp['Predicted_data']=ATrain_yPred_cls
ActT_Disp

TP_ATrain,TN_ATrain,FP_ATrain,FN_ATrain=TrueFalseValue(ActT_Disp,'Predicted_data','QC')
ConfMatATrain,tableATrain=confMatEval(TP_ATrain,TN_ATrain,FP_ATrain,FN_ATrain)
print(ConfMatATrain,'\n',tableATrain)


#Actual Test Prediction Evaluation Block

#Prediction for QC using the W for lowest J
ActTestPred=PredTest(x_ActTest,W_Atrain)
ActLocalTest['Predicted_data']=ActTestPred

TP_ATest,TN_ATest,FP_ATest,FN_ATest=TrueFalseValue(ActLocalTest,'Predicted_data','QC')
ConfMatSTest,tableSTest=confMatEval(TP_ATest,TN_ATest,FP_ATest,FN_ATest)
print(ConfMatSTest,'\n',tableSTest)
