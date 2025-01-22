import os 
from collections import defaultdict
import heapq 
import pickle 
import csv 
import math 

# used to preprocess Enron files for Polysys 
path_to_mail = "uk_gov_pay_junior"
db_list = []
output = defaultdict(list) 
output_length = defaultdict(int)

for filename in os.listdir(path_to_mail):
    print(filename)
    f = open(path_to_mail + "/" + filename,'r')
    csv_reader = csv.reader(f, delimiter=",")
    next(csv_reader, None)  # skip first line
    for line in csv_reader: 
        db_list.append(round(int(line[5]) * 0.01) - 217)

print(db_list)
print(len(db_list))
print(min(db_list))
print(max(db_list))

# with open(path_to_mail + "/test.csv", 'w',newline = '') as f: 
#     f_writer = csv.writer(f, delimiter=',')
#     for x in db_list: 
#         f_writer.writerow([x])

output = [[0 for _ in range(max(db_list))] for _ in range(len(db_list))]

for i in range(len(db_list)): 
    print(i, db_list[i])

    output[i][db_list[i] - 1] = 1

# Pickle the matrix
with open('salaries.pkl', 'wb') as f:
    pickle.dump(output, f)