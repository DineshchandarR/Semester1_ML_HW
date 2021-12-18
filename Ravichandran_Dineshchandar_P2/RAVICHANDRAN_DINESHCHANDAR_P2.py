# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 13:18:10 2021

@author: Dineshchandar Ravichandran,C19657741

Course: CPSC-6430-002-91763

Prof: Dr.Nianyi Li

Project: Linear Regression
"""

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import re
from sklearn import linear_model

print("====Enter Details For Training Data====")

# Training file as user input
def fileUpload():
    global fname 
    fname = input('Welcome \nKindly enter the file path of .csv you would like to load for trainig or testing and press enter:\n{eg:"G:\File.csv"}\n:')
    fpathValid = "\.(csv)"
    #Validating file type as ".csv"
    if (re.search(fpathValid,fname)):
       flen = len(fname)
       global i
       i=0
       while True:
           if flen > 0 and i<=3:
               con = input("Confirm file path:"+fname+" \n Press 'Y' to continue and Press 'N' to try again\n:").lower()
               conf = con
               str(conf)
               if conf == 'y' and (re.search(fpathValid,fname)):
                   break
               elif conf =='y'and not (re.search(fpathValid,fname)):
                   print("Attepmt remaining", (3-i))
                   fname = input("Invalid file type(must contain '.csv'.\n Kindly re-enter file path\n:")
                   flen = len(fname)
                   i+=1    
               elif conf =='n'and i<=3:
                   print("Attepmt remaining", (3-i))
                   fname = input("Kindly re-enter file path\n:")
                   flen = len(fname)
                   i+=1    
               else:     
                    print ("Invaild input")
                    i+=1
                    break
           elif flen <=0 and i<=3:
                print ("File path cannot be empty,try again")
                print("Attepmt remaining:", (3-i))
                fname = input("Kindly re-enter file path:")
                flen = len(fname)
                i+=1
           else:
                print ("Sorry Reached Max Attempts!!")
                sys.exit()
                break
    else:
        retry=input("Invalid File Type, Want to try again? (Y/N) \n:").lower()
        retryTxt=retry
        str(retryTxt)
        j=0
        if retryTxt == 'y' and j<=3:
            fileUpload(fileUpload())
        else:
            print("Tankyou!!")    
            sys.exit()
        
    return fname

fileUpload()

try:
    def load_Data(fpath=fname):
        return pd.read_csv(fpath)
    TD= load_Data()
except FileNotFoundError:
    k=0
    if k<=3:
        print("File not found kindly, check the file you have uploaded")
        k+=1
        fileUpload()
    else:
        print("Sorry Reached Max Attempts!!")

#File Read

try:
    def load_Data(fpath=fname):
        return pd.read_csv(fpath)
    TD= load_Data()
except FileNotFoundError:
    k=0
    if k<=3:
        print("File not found kindly, check the file you have uploaded")
        k+=1
        fileUpload()
    else:
        print("Sorry Reached Max Attempts!!")

TD.info()
TD['rooms_per_household'] = TD['total_rooms']/TD['households']
TD.head()
TD.describe()

#Representation of data on Histogram
#TD.hist(bins=50, figsize=(20,15))
#plt.show()

#Calculation of Weight

'''
w = (XT*X)**-1* XTy

'''
def weight (X,Y,C):
    A = np.linalg.pinv(np.dot(X.T,X))
    B = np.dot(X.T ,Y)
    W = np.dot(A, B)
    print("Weight for",C,"is:",W)
    return W

def cost (X,Y,w,C):
    '''
    j(w0,w1)=(i/m)*(Xw-y)T*(Xw-y)
    '''
    m=len(X)
    D = np.dot(X, w)-Y
    J = (1/m)*np.dot(D.T, D)
    print("J Value For",C,"is:",J)
    return J

def x_val(X):
    X_mat=np.vstack((np.ones(len(X)), X)).T
    return X_mat

def predict(x_input,W,C):
    pred=np.dot(x_input,W)
    print("Predicted Value For",C,"is:",pred)
    return pred

#Scikit Learn Comparision
def sk_com(x_sk,y_sk):
    model=linear_model.LinearRegression()
    model.fit(x_sk,y_sk)
    sk_T=model.predict(x_sk)
    print(sk_T)
    return sk_T

#Comparison table between the predicted output and the ground truth labels function.
def Compare (x_com,y_com,pred,com,sk_d):
    predDF=pd.DataFrame(data=pred,columns=["Predicted_Values"])
    pred_data=predDF.Predicted_Values
    sk_dDF =pd.DataFrame(data=sk_d,columns=["Sk_Predicted_Values"])
    sk_d1 =sk_dDF.Sk_Predicted_Values
    train_error= ((abs(y_com-pred_data)/y_com)*100)
    compareDF= pd.DataFrame(data={'Rooms_per_household': x_com ,
                                'Median_house_value(GT)':y_com,
                                'Predicted_Vlaues':pred_data,
                                  'Error in %':train_error,
                                  'Sk_Prediction':sk_d1,
                                  
                               }
                            )
    print('Compare Data for',com,", here Ground Truth is represented by GT =\n",compareDF)
    return compareDF

#Compare Data Storage Function
def Store (data,data_type):
    path='F:/Clemson/COURSE/SEM-1/Machine Learning Implementation and Evaluation(CPSC6430)/Project 2/Ravichandran_Dineshchandar_P2/'+data_type+'.csv'
    data.to_csv(path_or_buf=path, sep=',', na_rep='', header=True, index=True)

#Trainig Data Flow
train_x_val=x_val(TD.rooms_per_household)
train_y_val=np.vstack(TD.median_house_value)
#Weight Function calling
weight_train= weight(train_x_val,train_y_val,"Training")
#Cost Function calling
J_train= cost(train_x_val,train_y_val,weight_train,"Training")
#Prediction Function calling
pred_train = predict(train_x_val,weight_train,"Training")
#Scikit Data Prediction Function Calling
sk_train=sk_com(TD[['rooms_per_household']],TD[['median_house_value']])
#Comparison Function Calling
compare_train= Compare(TD.rooms_per_household,TD.median_house_value,pred_train,"Training",sk_train)
#Storing of above table as CSV
Store (compare_train,'Traing')

#Testing DataFlow
#File Reading for Test

print("====Testing====")

fileUpload()

try:
    def load_Data(fpath=fname):
        return pd.read_csv(fpath)
    Test= load_Data()
except FileNotFoundError:
    k=0
    if k<=3:
        print("File not found kindly, check the file you have uploaded")
        k+=1
        fileUpload()
    else:
        print("Sorry Reached Max Attempts!!")

Test.info()
Test['rooms_per_household'] = TD['total_rooms']/TD['households']
Test.head()
Test.describe()

test_x_val=x_val(Test.rooms_per_household)
test_y_val=np.vstack(Test.median_house_value)

#Cost Function calling for test data
J_test= cost(test_x_val,test_y_val,weight_train,"Testing")
#Prediction Function calling for test data
pred_test = predict(test_x_val,weight_train,"Testing")
#Scikit function calling:
sk_test=sk_com(Test[['rooms_per_household']],Test[['median_house_value']])
#Compare Function Calling + Sumary:
print("=====Traing Summary======")
print("Weight: ",weight_train,"\n","\nj:",J_train)
print("====Test Summary=====")
print("Weight: ",weight_train,"\n","\nj:",J_test)
compare_test= Compare(Test.rooms_per_household,Test.median_house_value,pred_test,"Test",sk_test)
#Graphical Representation

plt.xlim(0,80)
plt.scatter(Test.rooms_per_household,Test.median_house_value,c="blue",label="rooms_per_household VS median_house_value")
plt.plot(Test.rooms_per_household,pred_test,c='red',label="Predicted Vlaue")
plt.title("Testing Data Prediction")
plt.ylabel('median_house_value', fontsize=14)
plt.xlabel('rooms_per_household', fontsize=14)
plt.legend()


#Saving the graph
plotname_test = "TestGraph.png"
plt.savefig(plotname_test, bbox_inches='tight')

Store (compare_test,'Testing')