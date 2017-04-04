#Scoring using Intersection/ Union
import numpy as np
from collections import defaultdict
import collections
import csv, os

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
            score= len(intersected)
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
          
path=os.getcwd()
data1='../word_clusters.csv'  
Manual_Cluster=ExtractCSVtoDICT(data1)
N = len(Manual_Cluster)

print '---Number of samples: ', N, '---'

for r in range(29,60): 
    data2='../Clustering_Gaussian/cluster_gaussian_reviews_' + str(r) + '.csv'
    #data3=os.path.join(path, "KmeansResults.csv")
    Gaussian_Cluster=ExtractCSVtoDICT(data2)
    #KMeans_Cluster=ExtractCSVtoDICT(data3)
#printMatrix2(scoring(Manual_Cluster,Gaussian_Cluster))
    scores1=scoring(Manual_Cluster,Gaussian_Cluster)
    #scores2=scoring(Manual_Cluster,KMeans_Cluster)

    Max_Scores = {}

    i = 0
    for row in scores1:
        scores = [float(score) for score in row]
        Max_Scores[i] = max(scores[:])
        i += 1

    purity_sum = 0
    #print Max_Scores
    for val in Max_Scores.values():
        purity_sum += val

    purity_score = float(purity_sum) / 391
    

    print "purity for gaussian clustering with %d clusters: %f" % (r, purity_score)

#np.savetxt(path+"/purity_gaussian.csv", scores1, delimiter=",")

#np.savetxt(path+"/purity_KMeans.csv", scores2, delimiter=",")




        
