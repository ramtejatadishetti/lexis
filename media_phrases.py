import pickle
import os, re, csv
import numpy 

pickle_file = '../../Semantria-embeddings/Results/result_300d_p_media.pickle'

file_name  = re.findall('result.*_p_(.*)\.pickle',pickle_file)
data       = pickle.load(open(pickle_file,'rb'))

phrases    = open(file_name[0] + '_phrases.csv','w')
phrase_writer = csv.writer(phrases, delimiter=',')

header = ["cluster ID", "Phrase"]
phrase_writer.writerow(header)

clusterfile1 = '../../Semantria-embeddings/media_optimal_cluster.pkl'
cluster1     = pickle.load(open(clusterfile1,'rb'))

phrasefile1  = '../../Semantria-embeddings/media_embeddings.pkl'
phrase1      = pickle.load(open(phrasefile1,'rb'))

print len(cluster1)
print len(phrase1)


for i in range(len(phrase1)):
    phrases = [cluster1[i], phrase1[i][0]]
    phrase_writer.writerow(phrases)




   
