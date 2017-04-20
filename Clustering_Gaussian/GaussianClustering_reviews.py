from sklearn import mixture
import os, csv
import pickle
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
from scipy import linalg
import numpy as np
#name of phrase is key, and value is vector
count=0
X=[]
Y=[]
#for filename in os.listdir(os.getcwd()):
    #print filename
a='../word2vec_gov_init_embedding/trained_averaged_embeddings.pickle'

#b=csv.reader(open('../bellagio.csv','rb'))

abc=pickle.load(open(a,'rb'))

val= type(abc.values())
#key=abc.keys()
for k,val in abc.items():
    if val == []: print k

#print len(abc.keys())
#print abc['entire trip']
#print key

for row in abc.keys():
    phrase = row
    #cid = row[0]  
    try:
        X.append(abc[phrase.lower()])
        #print phrase.lower(), len(abc[phrase.lower()])
        Y.append(phrase)
    except KeyError:
        print 'got a key error: ', phrase.lower()
    #print X
#pickle.load(things to pickle, file object)        
#gmm = mixture.GaussianMixture(n_components=5, covariance_type='full').fit(X)

print len(X)
print len(Y)
#Scores={}
#output = open('../bellagio_clusters/bellagio_gaussian.pickle', 'wb')
# Pickle dictionary using protocol 0.

'''
for v in range(2,20):
    gmm = mixture.GaussianMixture(n_components=v, covariance_type='full').fit(X)
    cluster = mixture.GaussianMixture.predict(gmm,X)
    #print cluster
    print "gmmcv"
    Scores[v]=silhouette_score(X, cluster, metric='euclidean',sample_size=1200)

print Scores    

Max=max(Scores.values())
n_clusters=0
pickle.dump(Scores, output)

for k in Scores.keys():
    if(Scores[k]==Max):
        n_clusters=k

print n_clusters        
'''

#n_components = input('number of clusters to be considered: ')

for n_cluster in range(4000,4001):
    
    print n_cluster
    file_obj = open('../word2vec_gov_init_embedding/gov_clusters_' + str(n_cluster) + '.csv','wb')
    out_file = csv.writer(file_obj,delimiter=',')
    gmm_1=mixture.GaussianMixture(n_components=n_cluster, covariance_type='full').fit(X)
    pred_clusters = mixture.GaussianMixture.predict(gmm_1,X)

    print pred_clusters
    for i in range(len(pred_clusters)):
        new_row = [pred_clusters[i],Y[i]]       
        out_file.writerow(new_row)        
    
    file_obj.close()
    

'''
color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])

print gmm_1.covariances_

def plot_results(X, Y, means, covariances, index, title):
    splot = plt.subplot(5, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-6., 4. * np.pi - 6.)
    plt.ylim(-5., 5.)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())

#plot_results(X, gmm_1.predict(X), gmm_1.means_, gmm_1.covariances_, 1,"Bayesian Gaussian mixture models with a Dirichlet process prior "r"for $\gamma_0=0.01$.")

'''

    
    

