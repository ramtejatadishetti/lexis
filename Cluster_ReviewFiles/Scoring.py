#Scoring using Intersection/ Union
import numpy as np
from collections import defaultdict
import collections
import csv

def ExtractCSVtoDICT(data):
    reader = csv.reader(open(data))
    Manual_Clusters=defaultdict(list)
    keys=[] 
    Values=[]
    for row in reader:
        keys.append(row[0])
        Values.append(row[1])
    i=0
    for i in range(0,len(keys)-1):
        k=keys[i]
        if(keys[i]==keys[i+1]):
            Manual_Clusters[k].append(Values[i])
        else:
            Manual_Clusters[k].append(Values[i])
            
    return Manual_Clusters

def scoring(c1,c2):
    n2=len(c2)
    n1=len(c1)
    Score_Matrix=np.arange((n2+1)*(n1+1))
    Score_Matrix = np.zeros((n2+1, n1+1))
    for keys in c2.keys():
        value2=c2[keys]
        for key in c1.keys():
            value1=c1[key]
            intersected=set(value1).intersection(value2)
            score= len(intersected)*1.0/len(value2)
            #zprint score
            Score_Matrix[int(keys)][int(key)]=round(score,3)
    return Score_Matrix

def printMatrix2(testMatrix):
    f = open(path+'matrix_output.txt', 'w')
    print ' ',
    for i in range(len(testMatrix[1])):
        print i,
    print
    for i, element in enumerate(testMatrix):
       saved_this=str(i), ' '.join(str(element))
       f.write(str(saved_this))  
       f.write("\n")

def column(matrix, i):
    return [row[i-1] for row in matrix]  
          
path="C:/Users/Shruti Jadon/Documents/Cluster_ReviewFiles/"
data1=path+"ManualClusters.csv"   
data2=path+"GaussianResults.csv"
data3=path+"KMeansResults.csv"

Manual_Cluster=ExtractCSVtoDICT(data1)
Gaussian_Cluster=ExtractCSVtoDICT(data2)
KMeans_Cluster=ExtractCSVtoDICT(data3)
#printMatrix2(scoring(Manual_Cluster,Gaussian_Cluster))
scores=scoring(Manual_Cluster,KMeans_Cluster)

np.savetxt(path+"foo_KMeans.csv", scores, delimiter=",")
   




        
