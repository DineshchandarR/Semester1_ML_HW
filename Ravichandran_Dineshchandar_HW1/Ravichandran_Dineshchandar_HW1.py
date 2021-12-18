# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 15:36:06 2021

@author: Dinesh R

Course: CPSC-6430-002-91763

Prof: Dr.Nianyi Li

HW1: KFold Implimentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

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
        #sep="\t", header=None,names=['sepal length','sepal width','petal length','petal width','species'],skiprows=0
        return pd.read_csv(fname,sep="\t",header=None,names=["Year","time"],skiprows=1 )
except FileNotFoundError:
    j=0
    if j<=3:
        print("File not found kindly, check the file you have uploaded")
        j+=1
        fileUpload()
    else:
        print("Sorry Reached Max Attempts!!")

Data=loadData()
Data

def D2Mat(x):
    X_mat=np.vstack((np.ones(len(x)),x)).T
    return X_mat
def verStack(y):
    Y_vec=np.vstack(y)
    return Y_vec
x_data,y_data=D2Mat(Data.Year),verStack(Data.time)
print(x_data,y_data)

#len(x_data)
def xPower(x_data,p):
    xparr=[]
    for i in range(len(x_data)):
        xp=(x_data[i])**p
        xparr.append(xp)    
    xparr=np.array(xparr)
    return xparr
x2arr=xPower(x_data,2)
x2arr=np.delete(x2arr,(0),axis=1)
x3arr=xPower(x_data,3)
x3arr=np.delete(x3arr,(0),axis=1)
print(x2arr,'\n',x3arr)

QuadMat= np.hstack((x_data,x2arr))
CubMat= np.hstack((QuadMat,x3arr))
print(QuadMat,"\n",CubMat)

# Weight calculation
def weight (X,Y,C):
    A = np.linalg.pinv(np.dot(X.T,X))
    B = np.dot(X.T ,Y)
    W = np.dot(A, B)
    print("Weight for",C,"is:",W)
    return W

# Cost Calculation
def cost (X,Y,w,C,C2):
    '''
    j(w0,w1)=(i/m)*(Xw-y)T*(Xw-y)
    '''
    m=len(X)
    D = np.dot(X, w)-Y
    J = (1/m)*np.dot(D.T, D)
    print("J Value For",C2,"with optional classifer",C,"is:",J,"And M:",m)
    return J

#Prediction
def predict(x_input,W,C):
    pred=np.dot(x_input,W)
    print("Predicted Value For",C,"is:",pred)
    return pred

w_linear_com=weight(x_data,y_data,"Liner Equation")
w_quad_com=weight(QuadMat,y_data,"Quadratic Equation")
w_cube_com=weight(CubMat,y_data,"Cubic Equation")

c_linear_com=cost(x_data,y_data,w_linear_com,"Liner Equation","Whole Data Set")
c_quad_com=cost(QuadMat,y_data,w_quad_com,"Quadratic Equation","Whole Data Set")
c_cube_com=cost(CubMat,y_data,w_cube_com,"Cubic Equation","Whole Data Set")

#Spliting the data-set into k folds:
def CVS(dataset,fold,C):
    arr=np.array(dataset)
    data_split=np.array_split(arr,fold)
    print("The SplitData for",C,"is:\n",data_split)
    return data_split
LXData=CVS(x_data,5,"Linear Eq X")
LXData=np.array(LXData)
LYData=CVS(y_data,5,"Linear Eq Y")
LYData=np.array(LYData)
QXData=CVS(QuadMat,5,"Quad Eq X")
QXData=np.array(QXData)
QYData=CVS(y_data,5,"Quad Eq Y")
QYData=np.array(QYData)
CXData=CVS(CubMat,5,"Cubic Eq X")
CXData=np.array(CXData)
CYData=CVS(y_data,5,"Cubic Eq Y")
CYData=np.array(CYData)

#Linear Computing 
w_liner_list=[]
c_linear_list=[]
c_test_lin=[]
pred_list_lin=[]
#try:
for i in range (len(LXData)):
    x_arr_copy=LXData
    y_arr_copy=LYData
    print(i)
    x_test=x_arr_copy[i]
    y_tets=y_arr_copy[i]
    x_arr_copy=np.delete(x_arr_copy,[i])
    x_arr_copy=np.concatenate(x_arr_copy)
    x_arr_copy=x_arr_copy.reshape(-1,2)
    y_arr_copy=np.delete(y_arr_copy,[i])
    y_arr_copy=np.concatenate(y_arr_copy)
    y_arr_copy=y_arr_copy.reshape(-1,1)
    w_linear=weight(x_arr_copy,y_arr_copy,i)
    w_liner_list.append(w_linear)
    p_liner=predict(x_arr_copy,w_linear,i)
    pred_list_lin.append(p_liner)
    c_linear=cost(x_arr_copy,y_arr_copy,w_linear,i,'train')
    c_linear_list.append(c_linear)
    c_lin_test=cost(x_test,y_tets,w_linear,i,'test')
    c_test_lin.append(c_lin_test)

#Quadratic Computing    
w_quad_list=[]
c_quad_list=[]
c_test_quad=[]
#try:
for i in range (len(QXData)):
    x_arr_copy=QXData
    y_arr_copy=QYData
    print(i)
    x_test=x_arr_copy[i]
    y_tets=y_arr_copy[i]
    x_arr_copy=np.delete(x_arr_copy,[i])
    x_arr_copy=np.concatenate(x_arr_copy)
    x_arr_copy=x_arr_copy.reshape(-1,3)
    y_arr_copy=np.delete(y_arr_copy,[i])
    y_arr_copy=np.concatenate(y_arr_copy)
    y_arr_copy=y_arr_copy.reshape(-1,1)
    w_quad=weight(x_arr_copy,y_arr_copy,i)
    w_quad_list.append(w_quad)
    c_quad=cost(x_arr_copy,y_arr_copy,w_quad,i,'Train')
    c_quad_list.append(c_quad)
    c_quad_test=cost(x_test,y_tets,w_quad,i,"Test")
    c_test_quad.append(c_quad_test)

#Cubic Computing 
w_cube_list=[]
c_cube_list=[]
c_test_cube=[]
#try:
for i in range (len(CXData)):
    x_arr_copy=CXData
    y_arr_copy=CYData
    print(i)
    x_test=x_arr_copy[i]
    y_tets=y_arr_copy[i]
    x_arr_copy=np.delete(x_arr_copy,[i])
    x_arr_copy=np.concatenate(x_arr_copy)
    x_arr_copy=x_arr_copy.reshape(-1,4)
    y_arr_copy=np.delete(y_arr_copy,[i])
    y_arr_copy=np.concatenate(y_arr_copy)
    y_arr_copy=y_arr_copy.reshape(-1,1)
    w_cube=weight(x_arr_copy,y_arr_copy,i)
    w_cube_list.append(w_cube)
    c_cube=cost(x_arr_copy,y_arr_copy,w_cube,i,'Train')
    c_cube_list.append(c_cube)
    c_cube_test=cost(x_test,y_tets,w_cube,i,'Test')
    c_test_cube.append(c_cube_test)
    
summaryDF=pd.DataFrame(data={
                                "Fold":['2345','1-Test','1345','2-Test','1245','3-test','1235','4-Test','1234','5-Test','Mean Train','Mean Test'],
                                "Linear J":[c_linear_list[0],c_test_lin[0],c_linear_list[1],c_test_lin[1],c_linear_list[2],c_test_lin[2],c_linear_list[3],c_test_lin[3],c_linear_list[4],c_test_lin[4],np.mean(c_linear_list),np.mean(c_test_lin)],
                               "Quadratic J":[c_quad_list[0],c_test_quad[0],c_quad_list[1],c_test_quad[1],c_quad_list[2],c_test_quad[2],c_quad_list[3],c_test_quad[3],c_quad_list[4],c_test_quad[4],np.mean(c_quad_list),np.mean(c_test_quad)],
                                "Cubic J":[c_cube_list[0],c_test_cube[0],c_cube_list[1],c_test_cube[1],c_cube_list[2],c_test_cube[2],c_cube_list[3],c_test_cube[3],c_cube_list[4],c_test_cube[4],np.mean(c_cube_list),np.mean(c_test_cube)],
                            })
print(summaryDF)

summaryDF.to_csv(path_or_buf="F:/Clemson/COURSE/SEM-1/Machine Learning Implementation and Evaluation(CPSC6430)/HW1/HW1.csv")

quadPred=predict(QuadMat,w_quad_com,'Quadratic Prediction')

quadPred=predict(QuadMat,w_quad_com,'Quadratic Prediction')
linPred=predict(x_data,w_linear_com,'Linear Prediction')
cubPred=predict(CubMat,w_cube_com,'Cubic Prediction')

#Graphical Representation
def jGraph(j):
    Jarr = np.array(j)
    Jarr = Jarr.reshape(-1,1)
    return Jarr

J_lin_train = jGraph(c_linear_list)
J_lin_test = jGraph(c_test_lin)
J_quad_train = jGraph(c_quad_list)
J_quad_test = jGraph(c_test_quad)
J_cube_train = jGraph(c_cube_list)
J_cube_test = jGraph(c_test_cube)



plot1 = plt.figure(1)
plt.scatter(Data.Year,Data.time,c="blue",label="Ground Truth")
plt.plot(Data.Year,linPred,c='green',label="Predicted Linear Vlaue")
plt.plot(Data.Year,quadPred,c='red',label="Predicted Quadratic Vlaue")
plt.plot(Data.Year,cubPred,c='black',label="Predicted Cubic Vlaue")
plt.title("Women's Olympics 100m")
plt.ylabel('Times in seconds', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.legend()

plot2 = plt.figure(2)
y_axis=[1,2,3,4,5]
plt.plot(y_axis,J_lin_train,c='Blue',marker="o", label="Liner Train J vlaue")
plt.plot(y_axis,J_lin_test,c='Orange',marker="o", label="Liner Test J vlaue")
plt.title("Liner Model Jtest vs Jtrain")
plt.ylabel('J_Value', fontsize=14)
plt.xlabel('Fold', fontsize=14)
plt.legend()

plot3 = plt.figure(3)
y_axis=[1,2,3,4,5]
plt.plot(y_axis,J_quad_train,c='Blue',marker="o", label="Liner Train J vlaue")
plt.plot(y_axis,J_quad_test,c='Orange',marker="o", label="Liner Test J vlaue")
plt.title("Quadratic Model Jtest vs Jtrain")
plt.ylabel('J_Value', fontsize=14)
plt.xlabel('Fold', fontsize=14)
plt.legend()

plot4 = plt.figure(4)
y_axis=[1,2,3,4,5]
plt.plot(y_axis,J_cube_train,c='Blue',marker="o", label="Liner Train J vlaue")
plt.plot(y_axis,J_cube_test,c='Orange',marker="o", label="Liner Test J vlaue")
plt.title("Cubic Model Jtest vs Jtrain")
plt.ylabel('J_Value', fontsize=14)
plt.xlabel('Fold', fontsize=14)
plt.legend()
plt.show()

plot5 = plt.figure(5)
xaxis=[1,2,3]
j_test_mean = [np.mean(c_linear_list), np.mean(c_quad_list), np.mean(c_cube_list)]
j_train_mean= [np.mean(c_test_lin),np.mean(c_test_quad),np.mean(c_test_cube)] 
plt.plot(xaxis,j_test_mean,c='Blue',marker="o", label="Liner Train J vlaue")
plt.plot(xaxis,j_train_mean,c='Orange',marker="o", label="Liner Test J vlaue")
plt.ylabel('Squared Error Cost Function', fontsize=14)
plt.xlabel('Highest Polynomial Degree', fontsize=14)
plt.legend()
plt.show()


#Saving the graph
plotname_test = "HWGraph.png"
plt.savefig(plotname_test, bbox_inches='tight')
