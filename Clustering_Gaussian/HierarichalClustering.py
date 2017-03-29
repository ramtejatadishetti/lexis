from sklearn import mixture
import os
import pickle
from sklearn.metrics import silhouette_score
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import csv
#name of phrase is key, and value is vector
count=0
X=[]
Y=[]
all_files={}
for filename in os.listdir('C:/Users/Shruti Jadon/Documents/Semantria-embeddings/'):
    a="C:/Users/Shruti Jadon/Documents/Semantria-embeddings/"+filename
    abc=pickle.load(open(a,'rb'))
    val= abc.values()
    key=abc.keys()
    k= ''.join(key)
    for v in val:  
        X.append(v)
    for k in key:
        Y=np.append(Y,str(k))
        
    clusters={}  
    Scores={}
    # Pickle outputfile

    output_1 = open('C:/Users/Shruti Jadon/Documents/clustering_output/Gau'+filename+'.pkl', 'w+')
    
    gmm_1 = mixture.GaussianMixture(n_components=1500, covariance_type='full').fit(X)    
    #pickle.dump(gmm_1, output_1)          
    #fuck= gmm_1.predict(X, y=None)
 
    ofile  = open('C:/Users/Shruti Jadon/Documents/clustering_output/csv'+filename+'.csv', "wb")
    writer = csv.writer(ofile, delimiter=',')
    for k in abc.keys():
         v= abc.get(k)
         clusters=[]  
         cluster_num=gmm_1.predict(v,y=None)
         clusters.append(cluster_num[0])
         clusters.append(k)
         writer.writerow(clusters)
"""

    for v in range(1200,1700):
        gmm = mixture.GaussianMixture(n_components=v, covariance_type='full').fit(X)
        Y=gmm.predict(X, y=None)
        print v
        Scores[v]=silhouette_score(X, Y, metric='cosine',sample_size=1200)
        
    print Scores   
    Max=max(Scores.values())
    for k in Scores.keys():
        if(Scores[k]==Max):
            n_clusters=k
            
    print "number of clusters will be"
    print (n_clusters)     
"""