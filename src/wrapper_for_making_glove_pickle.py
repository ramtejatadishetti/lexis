import os
import pickle
import numpy as np

#Directory path to input glove directory
path_dir = "/media/tramteja/Windows/Users/ramteja/Documents/Spring-17/Nlp/extracted_glove_dir/"

# Function to read data from txt file , parse and return a dictionary
def loadvec(file):
	glove_dict = {} 
	fin= open(file)    
	for line in fin:
		items = line.replace('\r','').replace('\n','').split(' ')
		if len(items) < 10: continue
		word = items[0]
		vect = np.array([float(i) for i in items[1:] if len(i) > 1])
		glove_dict[word] = vect
	return glove_dict

#input file paths
file_with_100_dimensions = "glove.6B.100d.txt"
file_with_200_dimensions = "glove.6B.200d.txt"
file_with_300_dimensions = "glove.6B.300d.txt"
file_with_50_dimensions = "glove.6B.50d.txt"

#output_file_paths
out_file_with_100_dimensions = "glove_100d.pickle"
out_file_with_200_dimensions = "glove_200d.pickle"
out_file_with_300_dimensions = "glove_300d.pickle"
out_file_with_50_dimensions = "glove_50d.pickle"

#get the dictionary and dump into a pickle
full_path = path_dir + file_with_100_dimensions
glove_100_dimensions = loadvec(full_path)

full_path = path_dir + out_file_with_100_dimensions
with open(full_path, 'wb') as handle:
    pickle.dump(glove_100_dimensions, handle, protocol=pickle.HIGHEST_PROTOCOL)

full_path = path_dir + file_with_200_dimensions
glove_200_dimensions = loadvec(full_path)

full_path = path_dir + out_file_with_200_dimensions
with open(full_path, 'wb') as handle:
    pickle.dump(glove_200_dimensions, handle, protocol=pickle.HIGHEST_PROTOCOL)


full_path = path_dir + file_with_300_dimensions
glove_300_dimensions = loadvec(full_path)

full_path = path_dir + out_file_with_300_dimensions
with open(full_path, 'wb') as handle:
    pickle.dump(glove_300_dimensions, handle, protocol=pickle.HIGHEST_PROTOCOL)


full_path = path_dir + file_with_50_dimensions
glove_50_dimensions = loadvec(full_path)

full_path = path_dir + out_file_with_50_dimensions
with open(full_path, 'wb') as handle:
    pickle.dump(glove_50_dimensions, handle, protocol=pickle.HIGHEST_PROTOCOL)


print "Done"
