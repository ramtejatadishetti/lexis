import os, re
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle

#load phrases as hyphenated words
with open("hyphenated_phrases_gov_data_set.pickle", 'rb') as handle:
    phrase_list = pickle.load(handle)



#List of files that needs to be processed
fileList = {};
datapath = "./Data"
datafolders = os.listdir(datapath);
for folder in datafolders:
    folderpath = os.path.join(datapath, folder);
    datafile   = os.listdir(folderpath);
    for files in datafile:
        if re.match(".*\.txt",files):
            filepath = os.path.join(folderpath, files);
            fileList[files] = filepath;


for entry in fileList.keys():
    print "Analysing file: " + fileList[entry]
    text = open(fileList[entry]).read().decode('utf8')
    text_tokens =  nltk.word_tokenize(text)
    #text_tokens = [nltk.word_tokenize(sent) for sent in sentences] 
    print len(text_tokens)

    final_list = ""
    i = 0
    while i < len(text_tokens):
        #print text_tokens[i]
        single_word = text_tokens[i].lower()
        
        if(i+1 < len(text_tokens) ):
            double_word = single_word + "-" + text_tokens[i+1].lower()

        if(i+2 < len(text_tokens) ):
         triple_word = double_word + "-" + text_tokens[i+2].lower()

        if(i+2 < len(text_tokens) ):
            if triple_word in phrase_list:
                final_list += triple_word + " "
                i += 3
                continue

        if(i+1 < len(text_tokens) ):
            if double_word in phrase_list:
                final_list += double_word + " "
                i += 2
                continue

        final_list += single_word + " "
        i += 1

    
    out_path = "./documents_with_phrases/" + "result_" + entry
    #out_path.replace(".txt", "")

    target = open(out_path, 'w')
    target.write(final_list.encode("utf8"))
    target.close()


    


        
        
        



