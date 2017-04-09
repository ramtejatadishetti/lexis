import os,re
import pickle
import csv

#List of files that needs to be processed
fileList = {};
datapath = "./Phrases_semantria"
datafile = os.listdir(datapath);
for files in datafile:
    if re.match(".*\.csv",files):
        filepath = os.path.join(datapath, files);
        fileList[files] = filepath;

phrase_list = []
# make a pickle of phrases by reading
for entry in fileList.keys():
    with open(fileList[entry],'rb') as f:
        reader = csv.reader(f) 
        for row in reader:
            words_in_phrases = row[0].split()
            single_word = words_in_phrases[0].lower()
            if len(words_in_phrases) > 1:
                for i in range(1, len(words_in_phrases)):
                    single_word += '-'
                    single_word += words_in_phrases[i].lower()

            phrase_list.append(single_word)

result_name = "hyphenated_phrases_gov_data_set.pickle"
with open(result_name, 'wb') as handle:
    pickle.dump(phrase_list, handle, protocol=pickle.HIGHEST_PROTOCOL)







