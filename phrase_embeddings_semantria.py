import os, re, csv
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

path_to_glove_dir = "/home/amehta/"
miss_dict = {}
flag = 0

#load Glove dictionaries from pickle
with open(path_to_glove_dir + "glove_300d.pickle", 'rb') as handle:
  glove_300d_dict = pickle.load(handle)
  
def update_dict(word):
     flag = 1
     if word in miss_dict:
            miss_dict[word] = miss_dict[word] + 1
     else:
            miss_dict[word] = 1 
                   

def get_average_phrase_embedding_300d(phrase):
    list_of_words = phrase.split()
    init_embedding = np.zeros(300)
    phrase_length = len(list_of_words)
    flag = 0

    for i in range(0,len(list_of_words)):
        if list_of_words[i] in glove_300d_dict:
        # checking the length of embedding since some of the embeddings in glove are of lesslength example check embeddinglength assess
            if len(init_embedding) == len(glove_300d_dict[list_of_words[i]]) :
                init_embedding = init_embedding + glove_300d_dict[list_of_words[i]]
            else:
                #init_embedding = init_embedding + glove_300d_dict['<unk>']
                update_dict(list_of_words[i])
                break

        # if word is not present then try for its lemma
        else:
            print 'word not found, checking its lemma.'
            lemmatizer = WordNetLemmatizer()
            try:
              lemmatized_word = lemmatizer.lemmatize(list_of_words[i])  
              if lemmatized_word in glove_300d_dict:
                 if  len(init_embedding) == len(glove_300d_dict[lemmatized_word]) : 
                     init_embedding = init_embedding + glove_300d_dict[lemmatized_word]
                 else:
                     #init_embedding = init_embedding + glove_300d_dict['<unk>']
                     update_dict(list_of_words[i]) 
                     break    

        # if lemma is not present treat the word as unknown tag
              else:
                #init_embedding = init_embedding + glove_300d_dict['<unk>']
                 update_dict(list_of_words[i])
                 break
                         
            except UnicodeDecodeError:
                print "Unicode-decode error"  
                update_dict(list_of_words[i])
                break             

    if flag == 1:
       init_embedding = np.zeroes(300)
       return init_embedding
    else:   
       for i in range(0,len(init_embedding)):  
         init_embedding[i] = float(init_embedding[i])/phrase_length
         return init_embedding
        
datapath = './Phrases_semantria'
datafiles = os.listdir(datapath)

for f in datafiles:
    g_dict = {}
    filename = os.path.join(datapath, f)
    with open(filename,'rb') as semantria_file:
        semantria = csv.reader(semantria_file)
        for row in semantria:
            themes = row[0].split(',');
            phrase = themes[0].lower()
            embedding = get_average_phrase_embedding_300d(phrase)
            if np.all(embedding != 0):
               g_dict[phrase] = embedding
    write_file = open('result_300d_' + f.split('.')[0] + '.pickle','wb')
    pickle.dump(g_dict, write_file)
 
print "Number of missing words are: " , len(miss_dict)    
f = open('miss_phrases.txt','w')
f.write(str(miss_dict))
f.close()

  
