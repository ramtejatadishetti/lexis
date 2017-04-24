import numpy as np
import csv

feature_size = 10
vocabulary_size = 10
embedding_size = 300
batch_size = 1
num_sampled = 64    # Number of negative examples to sample.

global_dict_for_indices = {}
count_of_tokens ={}

g_phrase_dict = {}
g_vocab_dict = {}
g_reverse_vocab_dict = {}
g_train_dict = {}

INDEX_MIN = 0
INDEX_MAX = 24
CONTEXT_MIN = 25
CONTEXT_MAX = 28

W1_INDEX = 0
W2_INDEX = 25
C1_INDEX = 50
C2_INDEX = 51
C3_INDEX = 52
C4_INDEX = 53

INDEX_START = 0
INDEX_END = 50

def build_phrase_records(input_csv_records, phrase_dict):
    total_phrase_count = 0
    for i in range(0,len(input_csv_records)):
        if i%2 == 1:
            phrase_record = []
            for j in range(INDEX_MIN, INDEX_OUT+1):
                phrase_record.append(input_csv_records[i-1][j])
            
            for j in range(INDEX_MIN, INDEX_OUT+1):
                phrase_record.append(input_csv_records[i][j])

            for j in range(CONTEXT_MIN, CONTEXT_MAX+1):
                phrase_record.append(input_csv_records[i][j])
            
            phrase_dict[total_phrase_count] =  phrase_record
            total_phrase_count += 1

    return total_phrase_count


def make_vocab_index(total_phrase_count, phrase_dict, vocab_dict, reverse_vocab_dict):
    index_count = 0
    for i in range(0, total_phrase_count):
        if phrase_dict[i][W1_INDEX] not in vocab_dict:
            vocab_dict[phrase_dict[i][W1_INDEX] ] = index_count
            reverse_vocab_dict[index_count] = phrase_dict[i][W1_INDEX]
            index_count += 1
        
        if phrase_dict[i][W2_INDEX] not in vocab_dict:
            vocab_dict[phrase_dict[i][W2_INDEX] ] = index_count
            reverse_vocab_dict[index_count] = phrase_dict[i][W2_INDEX]
            index_count += 1
        
        if phrase_dict[i][C1_INDEX] not in vocab_dict:
            vocab_dict[phrase_dict[i][C1_INDEX] ] = index_count
            reverse_vocab_dict[index_count] = phrase_dict[i][C1_INDEX]
            index_count += 1
        
        if phrase_dict[i][C2_INDEX] not in vocab_dict:
            vocab_dict[phrase_dict[i][C2_INDEX] ] = index_count
            reverse_vocab_dict[index_count] = phrase_dict[i][C2_INDEX]
            index_count += 1
        
        if phrase_dict[i][C3_INDEX] not in vocab_dict:
            vocab_dict[phrase_dict[i][C3_INDEX] ] = index_count
            reverse_vocab_dict[index_count] = phrase_dict[i][C3_INDEX]
            index_count += 1
        
        if phrase_dict[i][C4_INDEX] not in vocab_dict:
            vocab_dict[phrase_dict[i][C4_INDEX] ] = index_count
            reverse_vocab_dict[index_count] = phrase_dict[i][C4_INDEX]
            index_count += 1
    
    return index_count

def make_valid_examples(total_phrase_count, phrase_dict, train_dict):
    train_count = 0
    for i in range(0, total_phrase_count):
        for k in range(0,4):
            phrase_record = []
            for j in range(INDEX_START, INDEX_END):
                phrase_record.append(phrase_dict[i][j])
            phrase_record.append(phrase_dict[i][INDEX_END + k])
            train_dict[train_count] = phrase_record
            train_count += 1
    
    return train_count

input_csv_list = []
with open('file.csv', 'rb') as f:
    reader = csv.reader(f)
    input_csv_list = list(reader)

total_phrase_count = build_phrase_records(input_csv_list, g_phrase_dict)
print "TOTAL_PHRASES", total_phrase_count

index_count = make_vocab_index(total_phrase_count, g_phrase_dict, g_vocab_dict, g_reverse_vocab_dict)
print "TOTAL VOCAB", index_count

train_count =  make_valid_examples(total_phrase_count, g_phrase_dict, g_train_dict)
print "TOTAL TRAIN EXAMPLES", train_count

with open("phrase.pickle", 'wb') as handle:
        pickle.dump(g_phrase_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
print "phrase_pickle completed"

with open("vocab.pickle", 'wb') as handle:
        pickle.dump(g_vocab_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
print "vocab_pickle completed"

with open("reverse_vocab.pickle", 'wb') as handle:
        pickle.dump(g_reverse_vocab_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
print "reverse_vocab_pickle completed"

with open("train.pickle", 'wb') as handle:
        pickle.dump(g_train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
print "train_pickle completed"

#load glove pickle
glove_300d_dict = None
with open(path_to_glove_dir + "glove_300d.pickle", 'rb') as handle:
    glove_300d_dict = pickle.load(handle)


embedding_size = 300
def get_word_embedding(unk, glove, word):
    if word in glove:
        return glove[word]
    elif word in unk:
        return unk[word]
    else:
        unk[word] = ( (2 )*np.random.rand(1, embedding_size) - 1 )
        glove[word] = unk[word]
        return glove[word]

#get embeddings for words in vocab
vocab_count = len(g_reverse_vocab_dict.keys())
unk_embeddings = {}

embeddings = np.ndarray(shape=(vocab_count, dim_size), dtype=np.float)
for i in range(0, vocab_count):
    word  = g_reverse_vocab_dict[i]
    word_embedding = get_word_embedding(unk_embeddings, glove_300d_dict, word)
    embeddings[i] = np.asarray(word_embedding)

with open("embeddings_reviews.pickle", 'wb') as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
print "embeddings_pickle completed"
