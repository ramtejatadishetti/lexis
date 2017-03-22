import os, re, pickle
import csv

pickled_embeddings  =  './tsne_embeddigns.pickle'
text_phrases        =  './media_phrases.csv'

embeddings_object   =  open(pickled_embeddings, 'rb')
phrase_object       =  open(text_phrases, 'rb')

media_embeddings    =  pickle.load(embeddings_object)
media_phrases       =  csv.reader(phrase_object, delimiter = ',')

#print "----Example of embedding and phrase----"
#print "phrase: ", media_phrases[0]
#print "embedding dimension 1: ", media_embeddings[0][0]
#print "embedding dimension 2: ", media_embeddings[0][1]


csv_object  =  open("media_plot.csv", "w")
media_csv    =  csv.writer(csv_object, delimiter = ",")

header      =  ["Cluster ID", "Phrase", "Dim1", "Dim2"]  

media_csv.writerow(header)

i = 0

for row in media_phrases:
    if row[0] != "cluster ID":
        csv_row   =  [row[0], row[1], media_embeddings[i][0], media_embeddings[i][1]]
        media_csv.writerow(csv_row)
        i += 1




