# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 00:05:11 2021

@author: Dineshchandar Ravichandran

Course: CPSC-6430-002-91763

Prof: Dr.Nianyi Li

P4: K-means clustering algorithm

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


#File upload and verification function
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
        return pd.read_csv(fname,sep="\t",header=None,names=["X","Y"],skiprows=1)

except FileNotFoundError:
    j=0
    if j<=3:
        print("File not found kindly, check the file you have uploaded")
        j+=1
        fileUpload()
    else:
        print("Sorry Reached Max Attempts!!")
        
#KMeans Algorithim

def KMean(Cent1,Cent2):
    #Euclidean dist
    distC1= np.sqrt((data['X']-Cent1[0])**2+(data['Y']-Cent1[1])**2)
    distC2= np.sqrt((data['X']-Cent2[0])**2+(data['Y']-Cent2[1])**2)
    
    #Nearest Centiod operation
    conditions = [ distC1 < distC2, distC1 > distC2]
    clusterClass = ['cluster1', 'cluster2']
    
    # Where distC1 < distC2 assgin class=Class1 and distC1 > distC2 assgin class=Class2
    data['cluster'] = np.select(conditions, clusterClass, default="NA")
    dfCluster1 = data[data['cluster'] == 'cluster1']
    #print(dfCluster1)
    dfCluster2 = data[data['cluster'] == 'cluster2']
    #print(dfCluster2)
    
    #Cost calculation
    #formula 1/m* (√((x_2-x_1)²+(y_2-y_1)²)**2
    J1 = np.sum((np.sqrt((dfCluster1['X'] - Cent1[0]) ** 2 + (dfCluster1['Y'] - Cent1[1]) ** 2))**2)
    J2 = np.sum((np.sqrt((dfCluster2['X'] - Cent2[0]) ** 2 + (dfCluster2['Y'] - Cent2[1]) ** 2))**2)
    J = (J1+J2)/len(data)
    print('\nError J-Value=',J)
    
    #Updation of Centiod based on mean of all points in cluster
    newCent1=[0,0]
    newCent2=[0,0]
    newCent1[0], newCent1[1] = round(sum(dfCluster1['X']) / len(dfCluster1),5), round(sum(dfCluster1['Y']) / len(dfCluster1),5)
    newCent2[0], newCent2[1] = round(sum(dfCluster2['X']) / len(dfCluster2),5), round(sum(dfCluster2['Y']) / len(dfCluster2),5)
    #newCent1[0], newCent1[1] = (sum(dfCluster1['X']) / len(dfCluster1)), (sum(dfCluster1['Y']) / len(dfCluster1))
    #newCent2[0], newCent2[1] = (sum(dfCluster2['X']) / len(dfCluster2)), (sum(dfCluster2['Y']) / len(dfCluster2))
    print('newCent1:',newCent1,'newCent2:',newCent2)
    
    # Visualising the clustered data
    plt.scatter(dfCluster1['X'],dfCluster1['Y'],c='green',marker="o")
    plt.scatter(newCent1[0],newCent1[1],c='black',marker="v")
    plt.scatter(dfCluster2['X'],dfCluster2['Y'],c='red',marker="o")
    plt.scatter(newCent2[0],newCent2[1],c='blue',marker="v")
    plt.title("Clustered Data points")
    plt.ylabel('X2 Axis')
    plt.xlabel('X1 Axis')
    plt.show()
    
    return newCent1,newCent2,J

#program initiation
if __name__=="__main__":
    print ("Enter the name of a Data file")
    data= loadData()
    print ("Enter the name of a Initial Centroid file")
    Centoid=loadData()
    
    # Visualising the intial data
    print('\nIntial Centeroids:\nCenteroid 1:',Centoid.iloc[0][0],',',Centoid.iloc[0][1],'\nCenteroid 2:',Centoid.iloc[1][0],',',Centoid.iloc[1][1])
    plt.scatter(data['X'],data['Y'],c='purple',marker="o")
    plt.scatter(Centoid['X'][0],Centoid['Y'][0],c='green',marker="v")
    plt.scatter(Centoid['X'][1],Centoid['Y'][1],c='red',marker="v")
    plt.title("Initial Data points")
    plt.ylabel('X2 Axis')
    plt.xlabel('X1 Axis')
    ogi=plt.show()
    print(ogi)
    
    Cent1, Cent2 = np.array(Centoid.iloc[0]),np.array(Centoid.iloc[1])
    itr = 100
    
    newCent1,newCent2,J = KMean(Cent1,Cent2)
    
    # Iteration untill dist btw old and new centiod is 0 or max itr is acchived.
    
    for i in range(itr):
        if newCent1[0] == Cent1[0] and newCent1[1] == Cent1[1] and newCent2[0] == Cent2[0] and newCent2[1] == Cent2[1]:
            print("\nFinal Centeriods1:",newCent1[0],newCent1[1],"\nFinal Centeriods2:",newCent2[0],newCent2[1])
            print("Final Error value is:",J)
            break
        else:
            print(i)
            Cent1,Cent2 = newCent1,newCent2
            newCent1,newCent2,J = KMean(Cent1,Cent2)