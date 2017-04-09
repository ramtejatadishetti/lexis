import os, re
import nltk
from nltk.stem import WordNetLemmatizer
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

gdict = {}
gdict1= {}
gdict2 = {}
gdict3 = {}
gdict_arbit = {}

posWordNet = {'NNP':'n','JJ':'a','NN':'n','PRP$':'n'}

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
            st = Lemmatizer.lemmatize(t[0][0].lower(),pos=posWordNet[t[0][1]])
            for i in range(1,len(t)):
                st += " " + Lemmatizer.lemmatize(t[i][0].lower(),pos=posWordNet[t[i][1]])

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

#fileList = [ path + "Comments_on_semiannual.txt"]
for entry in fileList.keys():
    print "Analysing file: " + fileList[entry]
    text = open(fileList[entry]).read().decode('utf8') 
    sentences =  nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences] 
    sentences = [nltk.pos_tag(sent) for sent in sentences]

    for sent in sentences:
        result = NPChunker.parse(sent)
        traverse(result)


    outfile = "./Results/" + "result_" + entry
    outfilep = open(outfile,"w")
    
    print "writing results to: " + outfile

    outfilep.write("\n Bigram NP \n")
    for w in sorted(gdict2, key=gdict2.get, reverse=True):
        outfilep.write((w + " " + str(gdict2[w]) + "\n").encode("utf8"))
        #print w, gdict2[w]


    outfilep.write("\n Trigram NP \n")
    for w in sorted(gdict3, key=gdict3.get, reverse=True):
        outfilep.write((w + " " + str(gdict3[w]) + "\n").encode("utf8"))
        #print w, gdict3[w]

    outfilep.write("\n Unigram NP \n")
    for w in sorted(gdict1, key=gdict1.get, reverse=True):
        outfilep.write((w + " " + str(gdict1[w]) + "\n").encode("utf8"))
        #print w, gdict1[w]


    outfilep.write("\n Arbitary length NP \n")
    for w in sorted(gdict_arbit, key=gdict_arbit.get, reverse=True):
        outfilep.write((w + " " + str(gdict_arbit[w]) + "\n").encode("utf8"))
        #print w, gdict_arbit[w]

    print "writing complete, saving file."

    gdict = {}
    gdict1= {}
    gdict2 = {}
    gdict3 = {}
    gdict_arbit = {}
    
    outfilep.close()


