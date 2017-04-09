import os
import re
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle

'''
Possible noun phrase patterns
'''
patterns = """
            NP: {<PRP\$>?<JJ>*<NN>}
                {<NNP>+}
                {<NN>+}
"""
NPChunker = nltk.RegexpParser(patterns)
Lemmatizer = WordNetLemmatizer()

path_to_glove_dir = "/media/tramteja/Windows/Users/ramteja/Documents/Spring-17/Nlp/extracted_glove_dir/"

# load Glove dictionaries from pickle
with open(path_to_glove_dir + "glove_50d.pickle", 'rb') as handle:
    glove_50d_dict = pickle.load(handle)

# with open(path_to_glove_dir + 'glove_100d.pickle', 'rb') as handle:
#  glove_100d_dict = pickle.load(handle)

# with open(path_to_glove_dir + 'glove_200d.pickle', 'rb') as handle:
#  glove_200d_dict = pickle.load(handle)

# with open(path_to_glove_dir + 'glove_300d.pickle', 'rb') as handle:
#  glove_300d_dict = pickle.load(handle)


def get_average_phrase_embedding_50d(phrase):
    list_of_words = phrase.split()
    init_embedding = np.zeros(50)
    phrase_length = len(list_of_words)

    for i in range(0, len(list_of_words)):
        if list_of_words[i] in glove_50d_dict:

            # checking the length of embedding since some of the embeddings in
            # glove are of lesslength example check embeddinglength assess
            if len(init_embedding) == len(glove_50d_dict[list_of_words[i]]):
                init_embedding = init_embedding + \
                    glove_50d_dict[list_of_words[i]]
            else:
                init_embedding = init_embedding + glove_50d_dict['<unk>']

        # if word is not present then try for its lemma
        else:
            lemmatized_word = Lemmatizer.lemmatize(list_of_words[i])
            if lemmatized_word in glove_50d_dict:
                if len(init_embedding) == len(glove_50d_dict[lemmatized_word]):
                    init_embedding = init_embedding + \
                        glove_50d_dict[lemmatized_word]
                else:
                    init_embedding = init_embedding + glove_50d_dict['<unk>']

        # if lemma is not present treat the word as unknown tag
            else:
                init_embedding = init_embedding + glove_50d_dict['<unk>']

    for i in range(0, len(init_embedding)):
        init_embedding[i] = float(init_embedding[i]) / phrase_length

    return init_embedding


gdict = {}
gdict1 = {}
gdict2 = {}
gdict3 = {}
gdict_arbit = {}

posWordNet = {'NNP': 'n', 'JJ': 'a', 'NN': 'n', 'PRP$': 'n'}

'''
Function to traverse a node in tree
'''


def traverse(t):
    try:
        t.label()
    except AttributeError:
        return

    else:
        if t.label() == 'NP':
            st = t[0][0].lower()
            for i in range(1, len(t)):
                st += " " + t[i][0].lower()

            if st in gdict:
                gdict[st] += 1
                if len(t) == 1:
                    gdict1[st] += 1
                elif len(t) == 2:
                    gdict2[st] += 1
                elif len(t) == 3:
                    gdict3[st] += 1
                else:
                    gdict_arbit[st] += 1
            else:
                gdict[st] = 1
                if len(t) == 1:
                    gdict1[st] = 1
                elif len(t) == 2:
                    gdict2[st] = 1
                elif len(t) == 3:
                    gdict3[st] = 1
                else:
                    gdict_arbit[st] = 1

        else:
            for child in t:
                traverse(child)


# List of files that needs to be processed
fileList = {}
datapath = "./Data"
datafolders = os.listdir(datapath)
for folder in datafolders:
    folderpath = os.path.join(datapath, folder)
    datafile = os.listdir(folderpath)
    for files in datafile:
        if re.match(".*\.txt", files):
            filepath = os.path.join(folderpath, files)
            fileList[files] = filepath

#fileList = [ path + "Comments_on_semiannual.txt"]
for entry in fileList.keys():
    print "Analysing file: " + fileList[entry]
    text = open(fileList[entry]).read().decode('utf8')
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]

    for sent in sentences:
        result = NPChunker.parse(sent)
        traverse(result)

    out_pickle_path = "./Results_pickle/" + "result_50d_" + entry
    out_pickle_path.replace(".txt", "")

    bigram_dict = {}
    trigram_dict = {}
    unigram_dict = {}
    multigram_dict = {}

    #outfilep = open(outfile,"w")

    # print "writing results to: " + outfile

    for w in sorted(gdict2, key=gdict2.get, reverse=True):
        bigram_dict[w] = get_average_phrase_embedding_50d(w)

    for w in sorted(gdict3, key=gdict3.get, reverse=True):
        trigram_dict[w] = get_average_phrase_embedding_50d(w)

    for w in sorted(gdict1, key=gdict1.get, reverse=True):
        unigram_dict[w] = get_average_phrase_embedding_50d(w)

    for w in sorted(gdict_arbit, key=gdict_arbit.get, reverse=True):
        multigram_dict[w] = get_average_phrase_embedding_50d(w)

    full_path = out_pickle_path + "_bigram.pickle"
    with open(full_path, 'wb') as handle:
        pickle.dump(bigram_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    full_path = out_pickle_path + "_trigram.pickle"
    with open(full_path, 'wb') as handle:
        pickle.dump(trigram_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    full_path = out_pickle_path + "_unigram.pickle"
    with open(full_path, 'wb') as handle:
        pickle.dump(unigram_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    full_path = out_pickle_path + "_multigram.pickle"
    with open(full_path, 'wb') as handle:
        pickle.dump(multigram_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print "writing complete, saving file."

    gdict = {}
    gdict1 = {}
    gdict2 = {}
    gdict3 = {}
    gdict_arbit = {}

    # outfilep.close()
