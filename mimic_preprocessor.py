import os 
from collections import defaultdict
import heapq 
import pickle 
import csv 
import math 

# used to preprocess Enron files for Polysys 
path_to_mail = "mimic"
db_list = []
output = defaultdict(list) 
output_length = defaultdict(int)

for filename in os.listdir(path_to_mail):
    print(filename)
    f = open(path_to_mail + "/" + filename,'r')
    csv_reader = csv.reader(f, delimiter=",")
    next(csv_reader, None)  # skip first line
    for line in csv_reader: 
        if int(line[3]) == 50995: 
            try: 
                db_list.append(round(float(line[5]) * 10))
            except: 
                if line[5] == "LESS THAN 0.1": 
                    db_list.append(1)
                elif line[5] == "<0.4": 
                    db_list.append(4) 
                elif line[5] == ">7.7" or "GREATER THAN 7.7": 
                    db_list.append(77)
                elif line[5] == "GREATER THAN 3.0": 
                    db_list.append(30) 
                elif line[5] == "<0.10":
                    db_list.append(1) 
                else: 
                    print("SHOULD NOT GET HERE")

# print(db_list)
print(len(db_list))
print(min(db_list))
print(max(db_list))

with open(path_to_mail + "/test.csv", 'w',newline = '') as f: 
    f_writer = csv.writer(f, delimiter=',')
    for i in range(500): 
        f_writer.writerow([db_list[i]])

output = [[0 for _ in range(max(db_list))] for _ in range(500)]

for i in range(500): 
    output[i][db_list[i] - 1] = 1

# Pickle the matrix
with open('mimic_t4.pkl', 'wb') as f:
    pickle.dump(output, f)