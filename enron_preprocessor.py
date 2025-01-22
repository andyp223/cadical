import os 
from collections import defaultdict
import heapq 
import pickle 

# used to preprocess Enron files for Polysys 
path_to_mail = "allen-p/_sent_mail"
output = defaultdict(list) 
output_length = defaultdict(int)
curr = 0
for filename in os.listdir("allen-p/_sent_mail"):
    f = open(path_to_mail + "/" + filename,'r')
    list_of_words = [word for line in f for word in line.split()]
    set_of_words = set(list_of_words)
    for word in set_of_words: 
        output[word].append(curr) 
        output_length[word] += 1
    curr += 1

top_100_keys = heapq.nlargest(220, output_length, key=output_length.get)

matrix = [[0 for _ in range(200)] for _ in range(602)]
for i in range(200): 
    key = top_100_keys[20+i] 
    for doc in output[key]: 
        print(doc)
        matrix[doc][i] = 1

# Pickle the matrix
with open('allen_p_matrix_200.pkl', 'wb') as f:
    pickle.dump(matrix, f)