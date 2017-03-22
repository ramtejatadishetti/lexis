import pickle
import os, re
import numpy 

pickle_files = './Semantria-embeddings'
filelist     = os.listdir(pickle_files)


for pickle_file in filelist:
    file_name  = re.findall('result.*_p_(.*)\.pickle',pickle_file)
    data       = pickle.load(open(os.path.join(pickle_files, pickle_file),'rb'))
    phrases    = open(file_name[0] + '_phrases.txt','a')
    embeddings = open(file_name[0] + '_p_embeddings.txt','a')
    array_write = []
    phrase_array = []
    for key in data.keys():
        phrase_array.append(key)
        phrases.write(key)
        phrases.write('\n')
        array_write.append(data[key])
    
    #print tuple(array_write)

    numpy.savetxt(embeddings, tuple(array_write))
    

    phrases.close()
    embeddings.close()

