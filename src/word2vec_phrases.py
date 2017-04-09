import os, re, numpy, pickle, csv
from gensim.models import word2vec


docs_path = './documents_with_phrases'

doc_list = os.listdir(docs_path)

sentences_all = []
line_count = 0
for doc in doc_list:
    doc_path = os.path.join(docs_path, doc)
    phrase_file = open(doc_path, 'rb')
    print "Analyzing file: ", doc
    for text in phrase_file:
        #sentences = text.split('.')
        line_count += 1
        print line_count
        #for sentence in sentences:
        sentence_tok=[w.lower() for w in text.split()]
        print sentence_tok
        sentences_all.append(sentence_tok)
        #if (line_count%1000 == 0): print "number of lines added: ", line_count


pickle.dump(sentences_all,open("sentences_for_wv","wb"))

word2vec_custom_model = word2vec.Word2Vec(sentences_all,min_count=1)
word2vec_custom_model.wv.save_word2vec_format('./word2vec_output')

pickle.dump(word2vec_custom_model,open('./phrase_word2vec.pickle','wb'))


semantria_phrases = pickle.load(open("hyphenated_phrases_gov_data_set.pickle","rb"))

phrase_embedding_dic = {}
count = 0
out_path = "./unknown_words.csv"
target = open(out_path, 'w')
wr = csv.writer(target, dialect='excel')

for phrase in semantria_phrases:
    try:
        embedding = word2vec_custom_model.wv[phrase]
        phrase_embedding_dic[phrase] = embedding
    except KeyError:
        count = count + 1
        wr.writerow([phrase])

print "Count of words not found :" , count
target.close()

pickle.dump(phrase_embedding_dic,open("./word2vec_phrase_embedding.pickle","wb"))

