from typing import List
from z3 import *
import pickle

from dataclasses import dataclass

import random
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import zipfian, randint

import operator
import functools

from typing import Any

def compute_recovery_rates(matrix,recovered_matrix): 
    r_number_of_records = len(matrix)
    n_domain_size = len(matrix[0]) 

    matrix_idx = [np.nonzero(x)[0][0] for x in matrix]
    recovered_matrix_idx = [np.nonzero(x)[0][0] for x in recovered_matrix]
    recovered_matrix_idx_reversed = [(n_domain_size - 1) - x for x in recovered_matrix_idx]

    # Compute approximation metric here
    approx_metric_left = sum([abs(i - j) for i,j in zip(matrix_idx, recovered_matrix_idx)])
    approx_metric_right = sum([abs(i - j) for i,j in zip(matrix_idx, recovered_matrix_idx_reversed)])

    approx_recovery_rate = 1 - min(approx_metric_left, approx_metric_right)/(math.floor(n_domain_size * r_number_of_records))

    return approx_recovery_rate

class Distribution:
    def pmf(self) -> List[RatNumRef]: ...
    def sample(self, number_of_queries: int) -> List[int]: ...

class Uniform(Distribution):
    def __init__(self, domain_size: int):
        self.domain_size = domain_size

    def pmf(self) -> List[RatNumRef]:
        return [Q(1, self.domain_size) for i in range(self.domain_size)]

    def sample(self, number_of_queries: int) -> List[int]:
        return randint.rvs(0, self.domain_size, size=number_of_queries)

class Zipfian(Distribution):
    def __init__(self, domain_size: int, s: int):
        self.domain_size = domain_size
        self.s = s

    def pmf(self) -> List[RatNumRef]:
        return [
            zipfian.pmf(i, self.s, self.domain_size, loc=-1)
            for i in range(self.domain_size)
        ]

    def sample(self, number_of_queries: int) -> List[int]:
        return zipfian.rvs(self.s, self.domain_size, size=number_of_queries, loc=-1)

def compute_amo_extra_info(n): 
    if n <= 4: 
        return 0, int(n * (n - 1) / 2)
    else: 
        extra_vars, extra_constraints = compute_amo_extra_info(n - 2)
        return 1 + extra_vars, 6 + extra_constraints

def compute_Q_info(t, n): 
    Q_vars = [[0 for _ in range(n)] for _ in range(t)]
    for i in range(t): 
        for j in range(n): 
            Q_vars[i][j] = (n * i) + j + 1

    Q_extra_vars = {} 
    for i in range(t): 
        Q_extra_vars[i] = {} 
        Q_extra_vars[i][0] = [t*n + (2*i) * n + j for j in range(1,n + 1) ] # after variables
        Q_extra_vars[i][1] = [t*n + (2*i + 1) * n + j for j in range(1,n + 1) ] # before variables

    return Q_vars, Q_extra_vars, 3*n*t, t* (5*n - 1)

def compute_D_info(t, r, n, curr_num): 
    D_vars = [[0 for _ in range(r)] for _ in range(n)]
    for i in range(n): 
        for j in range(r): 
            D_vars[i][j] = (r * i) + j + curr_num + 1
    curr_num += r*n 
    D_extra_vars = {} 
    extra_vars, extra_constraints = compute_amo_extra_info(n)
    for i in range(r):  
        D_extra_vars[i] = [curr_num + j for j in range(1, extra_vars + 1)]
        curr_num += extra_vars

    return D_vars, D_extra_vars, r*n + r*extra_vars, r * (extra_constraints + 1)

def compute_leakage_constraints(t,r,n,leakage,curr_num): 
    num_extra_vars = 0 
    num_extra_constraints = 0 

    rid_vars = {} 
    for i in range(t): 
        for j in range(r): 
            if leakage[i][j]: 
                rid_vars[(i,j)] = [curr_num + j + 1 for j in range(n)]
                curr_num += n 
                num_extra_vars += n 
                num_extra_constraints += 2*n + 1
            else: 
                rid_vars[(i,j)] = [] 
                num_extra_constraints += n

    return rid_vars,num_extra_vars, num_extra_constraints


def write_range_clauses_to_file(normal_vars, extra_vars, file): 
    n = len(normal_vars) 

    file.write(" ".join([str(x) for x in normal_vars]) + " 0\n")

    for i in range(n - 1): 
        file.write(str(-1 * extra_vars[0][i+1]) + " " + str(extra_vars[0][i]) + " 0\n")
        file.write(str(-1 * extra_vars[1][i]) + " " + str(extra_vars[1][i + 1]) + " 0\n")

    for i in range(n): 
        file.write(str(-1 * normal_vars[i]) + " " + str(extra_vars[0][i]) + " 0\n")
        file.write(str(-1 * normal_vars[i]) + " " + str(extra_vars[1][i]) + " 0\n")
        file.write(str(normal_vars[i]) + " " + str(-1 * extra_vars[0][i]) + " " + str(-1 * extra_vars[1][i]) + " 0\n")

def amo(column, extra_vars,file): 
    if len(extra_vars) == 0: 
        n = len(column) 
        for i in range(n): 
            for j in range(i + 1, n): 
                file.write(str(-1 * column[i]) + " " + str(-1 * column[j]) + " 0\n")
        return
    else: 
        assert(len(column) > 4)
        amo(column[:3] + [extra_vars[0]], [], file) 
        amo([-1 * extra_vars[0]] + column[3:],extra_vars[1:], file)

def write_pbeq_clause_to_file(column, extra_vars, file):
    n = len(column) 
    file.write(" ".join([str(x) for x in column]) + " 0\n")
    amo(column, extra_vars, file) 

def write_leakage_clauses_to_file(row,col,extra_vars,indicator_bit, file): 
    n = len(row)
    assert(len(row) == len(col)) 
    if indicator_bit: 
        assert(len(extra_vars) == len(row))
        file.write(" ".join([str(x) for x in extra_vars]) + " 0\n")
        for i in range(n): 
            file.write(str(-1 * extra_vars[i]) + " " + str(row[i]) + " 0\n")
            file.write(str(-1 * extra_vars[i]) + " " + str(col[i]) + " 0\n")
    else: 
        assert(len(extra_vars) == 0)
        for i in range(n): 
            file.write(str(-1 * row[i]) + " " + str(-1 * col[i]) + " 0\n")

def parse_output_file(D_matrix_mapping, r, n, num_vars):
    tmp = {} 
    output = [[0 for _ in range(n)] for _ in range(r)]
    with open('testing_rid.txt') as file: 
        while 1: 
            line = next(file).split(" ")
            if line[0] == 's': 
                if line[1] == "UNSATISFIABLE\n": 
                    return -1
                break
        line = next(file).split(" ")
        while line[0] == 'v': 
            for i in range(1, len(line)): 
                # print(int(line[i]))
                tmp[abs(int(line[i]))] = int(int(line[i]) > 0)
                if (int(line[i]) >= num_vars): 
                    break
            line = next(file).split(" ")

    for i in range(r): 
        for j in range(n): 
            if D_matrix_mapping[j][i] != 0: 
                output[i][j] = tmp[D_matrix_mapping[j][i]]
    return output

class RangeRidSolver:
    def __init__(
        self,
        t_number_of_queries: int,
        r_number_of_records: int,
        n_domain_size: int,
        file_name 
    ):
        self.t_number_of_queries = t_number_of_queries
        self.r_number_of_records = r_number_of_records
        self.n_domain_size = n_domain_size
        self.file_name = file_name 

    def run_iteration(self, leakage): 
        Q_vars, Q_extra_vars, num_Q_vars, num_Q_constraints = compute_Q_info(self.t_number_of_queries, self.n_domain_size)
        D_vars, D_extra_vars, num_D_vars, num_D_constraints = compute_D_info(self.t_number_of_queries, self.r_number_of_records, self.n_domain_size, num_Q_vars)
        rid_extra_vars,num_leakage_vars, num_leakage_constraints = compute_leakage_constraints(self.t_number_of_queries, self.r_number_of_records, self.n_domain_size, leakage, num_Q_vars+num_D_vars)

        f = open(self.file_name, 'w+') 
        print('p cnf ' + str(int(num_Q_vars + num_D_vars + num_leakage_vars)) + " " + str(int(num_Q_constraints + num_D_constraints + num_leakage_constraints)))
        f.write('p cnf ' + str(int(num_Q_vars + num_D_vars + num_leakage_vars)) + " " + str(int(num_Q_constraints + num_D_constraints + num_leakage_constraints)) + "\n") 

        # write the query matrix structure constraints 
        for i in range(self.t_number_of_queries): 
            write_range_clauses_to_file(Q_vars[i], Q_extra_vars[i], f)
        
        for i in range(self.r_number_of_records):
            write_pbeq_clause_to_file(np.array(D_vars)[:,i].tolist(), D_extra_vars[i], f)
        
        for i in range(self.t_number_of_queries):
            for j in range(self.r_number_of_records):
                row = [] 
                col = [] 
                for k in range(self.n_domain_size): 
                    row.append(Q_vars[i][k])
                    col.append(D_vars[k][j])
                    
                write_leakage_clauses_to_file(row,col,rid_extra_vars[(i,j)],leakage[i][j], f)

        f.close()
        # print(f"Elapsed: {(end - start) / (10 ** 9)} s")
        # print("Solving ...")
        cmd = './build/cadical ' +  self.file_name + ' > testing_rid.txt'
        os.system(cmd) 

        output = parse_output_file(D_vars, self.r_number_of_records, self.n_domain_size, num_Q_vars + num_D_vars)

        return output
    
    def solve(self, D_matrix, leakage):  
        start = time.perf_counter_ns() 
        recovered_D_matrix = self.run_iteration(leakage)
        # (recovered_D_matrix)

        end = time.perf_counter_ns() 
        runtime = (end - start) / (10 ** 9)

        recovery_rate = compute_recovery_rates(D_matrix, recovered_D_matrix)
        return runtime, recovery_rate 

def run_one_instance(t_number_of_queries, r_number_of_records, n_domain_size):
    # number of possible range queries
    number_of_ranges = n_domain_size *(n_domain_size + 1) /2

    # Step 1: generate the list of all possible ranges
    set_of_ranges = []
    for i in range(n_domain_size):
        for j in range(i, n_domain_size):
                set_of_ranges.append((i,j))
    number_of_ranges = len(set_of_ranges)

    queries_dist = Zipfian(number_of_ranges,2)
    #queries_dist = Uniform(number_of_ranges)
    sample = queries_dist.sample(t_number_of_queries)
    random_F = [i for i in range(number_of_ranges)]
    random.shuffle(random_F)
    queries = [random_F[q] for q in sample]

    data = Uniform(n_domain_size).sample(r_number_of_records)

    Q_matrix = [[0 for _ in range(n_domain_size)] for _ in range(t_number_of_queries)]
    D_matrix = [[0 for _ in range(n_domain_size)] for _ in range(r_number_of_records)]
    recovered_D_matrix = [[0 for _ in range(n_domain_size)] for _ in range(r_number_of_records)]

    for i in range(t_number_of_queries): 
        (lb,ub) = set_of_ranges[queries[i]]
        for j in range(lb,ub+1):
            Q_matrix[i][j] = 1
    
    for i in range(r_number_of_records): 
        recovered_D_matrix[i][data[i]] = 1
    
    with open('mimic_t4.pkl', 'rb') as f:
        data = pickle.load(f)

    matrix_idx = [np.nonzero(x)[0][0] for x in data]
    for i in range(len(matrix_idx)): 
        D_matrix[i][matrix_idx[i]] = 1

    L = np.matmul(Q_matrix, np.array(D_matrix).T.tolist()) 

    range_rid_solver = RangeRidSolver(
        t_number_of_queries=t_number_of_queries,
        r_number_of_records=r_number_of_records,
        n_domain_size=n_domain_size,
        file_name="test.cnf"
    )

    #data = Uniform(n_domain_size).sample(r_number_of_records)
    recovery_rate = compute_recovery_rates(D_matrix,recovered_D_matrix)
    # time_result, recovery_rate = range_rid_solver.solve(D_matrix, L)
    time_result = 0
    return time_result, recovery_rate

def main():
    f = open("range_rid_test_mimic_zipf.txt", "w")
    t_list = [100,150,200,250,300]
    r_list = [500]
    n_list = [64]
    num_iters = 5
    num_threads = 1

    for t_number_of_queries in t_list: 
        for r_number_of_records in r_list: 
            for n_domain_size in n_list: 
                with Pool(num_threads) as p: 
                    results = p.starmap(run_one_instance, [(t_number_of_queries,r_number_of_records,n_domain_size,) for _ in range(num_iters)])
                    averages = np.array(results).sum(axis=0) / num_iters
                    output = [t_number_of_queries, n_domain_size,averages[0], averages[1]]
                    print([t_number_of_queries, n_domain_size,averages[0], averages[1]])
                    f.write(",".join([str(x) for x in output]) + "\n")

if __name__ == "__main__":
    main()
    # n = 5 
    # k = 2
    # vec = [1,2,3,4,5]
    # s = [[0 for _ in range(k)] for _ in range(n)]
    # curr_num = 5
    # for i in range(n): 
    #     for j in range(k): 
    #         s[i][j] = curr_num + 1
    #         curr_num += 1 
    

    # file = open('test.cnf','w+')

    # k_out_of_n(vec,s,file)
