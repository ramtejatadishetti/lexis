import pickle
import os, re
import numpy

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


resultdatapath  = './Results'
resultdatafiles = os.listdir(resultdatapath)

default = raw_input('use default value for number of clusters? Press y or n: ')

if default == 'y': k = 25
elif default == 'n': 
    print 'You need to choose the maximum allowed clusters for data\n'
    k = input('input maximum number of desirable clusters: ')
else: 
    print 'Invalid entry. Using default setting'
    k = 25


for resultfile in resultdatafiles:
    phrase_embedding = []
    embeddings = []
    silhouettes = {}
    parentfile = re.findall('result_300d_p_(.*)\.pickle',resultfile)
    print parentfile[0]
    filepath    = os.path.join(resultdatapath, resultfile)
    phrase_file = pickle.load(open(filepath,'rb'))
    for phrase in phrase_file.keys():
        phrase_embedding.append((phrase, phrase_file[phrase]))
        embeddings.append(phrase_file[phrase])
                                    
    print 'number of phrases loaded: ', len(phrase_embedding)
    
    pickle.dump(phrase_embedding, open(parentfile[0] + '_embeddings.pkl','wb'))
    
    for n in range(2,k+1):
        print n
        model    = KMeans(n_clusters = n)
        clusters = model.fit_predict(embeddings)

        silhouettes[n] = silhouette_score(embeddings, clusters, sample_size = 1000)
    
    clusterfile = parentfile[0] + '_scores.pkl'    
    pickle.dump(silhouettes, open(clusterfile,'wb'))

    n_cluster = sorted(silhouettes, key=silhouettes.__getitem__, reverse=True)[0]

    print 'optimal number of clusters: ', n_cluster

    model    = KMeans(n_clusters = n_cluster)
    clusters = model.fit_predict(embeddings)
    
    optimal_cluster = parentfile[0] + '_optimal_cluster.pkl'
    pickle.dump(clusters, open(optimal_cluster,'wb'))


    print parentfile[0] + ' done. Moving on. \n'

'''
phrase_embeddings = []
silhouettes        = {}
for data in resultdatafiles:
    filepath    = os.path.join(resultdatapath, data)
    phrase_file = pickle.load(open(filepath,'rb'))
    for phrase in phrase_file.values():
        phrase_embeddings.append(phrase)

print 'number of phrases loaded: ', len(phrase_embeddings)

#Take the number of clusters as input. Higher number of clusters will result in more 
k = input('input maximum number of desirable clusters: ')

for n in range(2,k+1):

        model    = KMeans(n_clusters = n)
        clusters = model.fit_predict(phrase_embeddings)

        silhouettes[n] = silhouette_score(phrase_embeddings, clusters, sample_size = 1000)
        
pickle.dump(silhouettes, open('silhouette_scores.pkl','wb'))
'''
'''
silhouettes = pickle.load(open('silhouette_scores.pkl','rb'))
print silhouettes[2]

n_cluster = sorted(silhouettes, key=silhouettes.__getitem__, reverse=True)[0]

print 'optimal number of clusters: ', n_cluster

model    = KMeans(n_clusters = n_cluster)
clusters = model.fit_predict(phrase_embeddings)

pickle.dump(clusters, open('optimal_cluster.pkl','wb'))


clusters = pickle.load(open('optimal_cluster.pkl','rb'))

phrases = []

embeddings = []
for index, cluster in enumerate(clusters):
    for ind, phrase in enumerate(phrase_embeddings):
        if index == ind and cluster == 1: 
           print '------',  ind,  '------'
           embeddings.append(phrase)
    print index

pickle.dump(embeddings, open('embeddings','wb'))

embeddings = pickle.load(open('embeddings','rb'))
print embeddings
'''

