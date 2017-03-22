import pickle, os, re
import numpy

clusterfile1 = './Alcohol_Problems_optimal_cluster.pkl'
cluster1     = pickle.load(open(clusterfile1,'rb'))

phrasefile1  = './Alcohol_Problems_embeddings.pkl'
phrase1      = pickle.load(open(phrasefile1,'rb'))


print len(cluster1)
print len(phrase1)

phrases = {}

no_of_clusters = numpy.unique(cluster1)
print no_of_clusters
for j in no_of_clusters:
    new_list = []
    for i in range(len(cluster1)):
        if cluster1[i] == j: 
            #print i
            new_list.append(phrase1[i][0])
    phrases[j] = new_list  

print phrases[0][0:100]
