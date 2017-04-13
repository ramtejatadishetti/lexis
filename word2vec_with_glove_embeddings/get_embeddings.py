import numpy as np
import nltk
import pickle

from nltk.stem import WordNetLemmatizer
Lemmatizer = WordNetLemmatizer()



#open textfile
total_context = ""
file_name = 'tokenized_sentences.txt'

with open(file_name) as handle:
    total_context = handle.read().decode('utf8')


embeddings_dict = {}
sentences = nltk.sent_tokenize(total_context)

words_list = []

for sent in sentences:
    words_in_sentences = nltk.word_tokenize(sent)
    for word in words_in_sentences:
        words_list.append(word)


with open("glove_300d.pickle", 'rb') as handle:
    print("Opening pickle")
    glove_dict = pickle.load(handle)
    print('completed opening')



def get_average_phrase_embedding(phrase, embedding_size, low_index, high_index, unk_vecs):
    list_of_words = phrase.split('-')
    init_embedding = np.zeros(embedding_size)
    phrase_length = len(list_of_words)

    for i in range(0,len(list_of_words)):

        if list_of_words[i] in glove_dict:
            if len(init_embedding) == len(glove_dict[list_of_words[i]]):
                init_embedding = init_embedding + glove_dict[list_of_words[i]]
            else:
                if list_of_words[i] in unk_vecs:
                    init_embedding = init_embedding + unk_vecs[list_of_words[i]]

                else:
                    vec =  ( (high_index - low_index )*np.random.rand(1, embedding_size) + low_index )
                    unk_vecs[list_of_words[i]] = vec[0]
                    init_embedding = init_embedding + vec[0]



        # if word is not present then try for its lemma
        else:
            lemmatized_word = Lemmatizer.lemmatize(list_of_words[i])
            if lemmatized_word in glove_dict:
                if len(init_embedding) == len(glove_dict[lemmatized_word]):
                    init_embedding = init_embedding + glove_dict[lemmatized_word]

                else:
                    if list_of_words[i] in unk_vecs:
                        init_embedding = init_embedding + unk_vecs[list_of_words[i]]

                    else:
                        vec =  ( (high_index - low_index )*np.random.rand(1, embedding_size) + low_index )
                        unk_vecs[list_of_words[i]] = vec[0]
                        init_embedding = init_embedding + vec[0]


        # if lemma is not present treat the word as unknown tag
            else:
                if list_of_words[i] in unk_vecs:
                    init_embedding = init_embedding + unk_vecs[list_of_words[i]]
                else:
                    vec =  ( (high_index - low_index )*np.random.rand(1, embedding_size) + low_index )
                    unk_vecs[list_of_words[i]]  = vec[0]
                    init_embedding = init_embedding + vec[0]


    return init_embedding



def get_word_embeddings(data):
    
    print "Length of data", len(data)

    unk_vecs = {}
    final_embeddings = {}
    for i in range(0, len(data)):
        final_embeddings[data[i]] = get_average_phrase_embedding(data[i].lower() , 300, -1, 1, unk_vecs)

    return final_embeddings



embeddings = get_word_embeddings(words_list)

with open('gov_data_embeddings_final.pickle', 'wb') as handle:
    pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)





