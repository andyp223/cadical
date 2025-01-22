import os 
from collections import defaultdict
import heapq 
import pickle 
import csv 
import math 
from scipy.stats import randint

# used to preprocess Enron files for Polysys 
# path_to_mail = "uk_gov_pay_junior"
# db_list = []
# output = defaultdict(list) 
# output_length = defaultdict(int)

# for filename in os.listdir(path_to_mail):
#     print(filename)
#     f = open(path_to_mail + "/" + filename,'r')
#     csv_reader = csv.reader(f, delimiter=",")
#     next(csv_reader, None)  # skip first line
#     for line in csv_reader: 
#         db_list.append(round(int(line[5]) * 0.01) - 217)

n = 50 
r = 500
db_list = [] 
for i in range(n): 
    db_list.append(i) 

random_list = randint.rvs(0, n, size=r - n)

for i in range(r - n): 
    db_list.append(random_list[i])
# print(db_list)
# print(len(db_list))
# print(min(db_list))
# print(max(db_list))

with open("dense_uniform" + "/dense_uniform.csv", 'w',newline = '') as f: 
    f_writer = csv.writer(f, delimiter=',')
    for x in db_list: 
        f_writer.writerow([x])

output = [[0 for _ in range(max(db_list) + 1)] for _ in range(len(db_list))]

for i in range(len(db_list)): 
    print(i, db_list[i])

    output[i][db_list[i] - 1] = 1

# Pickle the matrix
with open('dense_uniform.pkl', 'wb') as f:
    pickle.dump(output, f)