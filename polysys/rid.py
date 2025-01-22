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

    # Compute approximation metric here
    approx_metric_left = sum([abs(i - j) for i,j in zip(matrix_idx, recovered_matrix_idx)])
    approx_recovery_rate = 1 - approx_metric_left/(math.floor(n_domain_size * r_number_of_records))

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
    extra_vars, extra_constraints = compute_amo_extra_info(n)
    for i in range(t):  
        Q_extra_vars[i] = [t*n + extra_vars * i + j for j in range(1, extra_vars + 1)]

    return Q_vars, Q_extra_vars, t*n + t*extra_vars, t* (extra_constraints + 1)

def compute_D_info(t, r, n, known_records, D_matrix, curr_num): 
    D_vars = [[0 for _ in range(r)] for _ in range(n)]
    num_vars = 0
    num_constraints = 0 

    for i in range(r): 
        if i in known_records: 
            for j in range(n): 
                D_vars[j][i] = D_matrix[i][j]
        else: 
            for j in range(n): 
                D_vars[j][i] = curr_num + 1 
                curr_num += 1 
                num_vars += 1

    # extra_vars, extra_constraints = compute_amo_extra_info(n)
    for i in range(r): 
        if i in known_records: 
            num_constraints += 0
            # D_extra_vars[i] = []
        else: 
            #  D_extra_vars[i] = [curr_num + j for j in range(1, extra_vars + 1)]

            # curr_num += extra_vars
            # num_vars += extra_vars 
            num_constraints += 1 

    return D_vars, num_vars, num_constraints

def compute_leakage_constraints(t,r,n,leakage,D_vars,curr_num): 
    num_extra_vars = 0 
    num_extra_constraints = 0 

    rid_vars = {} 
    for i in range(t): 
        for j in range(r): 
            var_nums = [D_vars[k][j] for k in range(n)]
            if leakage[i][j]: 
                count = len([x for x in var_nums if x > 1])
                if count > 0: 
                    rid_vars[(i,j)] = [curr_num + j + 1 for j in range(count)]
                    curr_num += count 
                    num_extra_vars += count 
                    num_extra_constraints += 2*count + 1
                else: 
                    rid_vars[(i,j)] = []
                    num_extra_constraints += 1
            else: 
                count = len([x for x in var_nums if x > 0])
                rid_vars[(i,j)] = [] 
                num_extra_constraints += count

    return rid_vars,num_extra_vars, num_extra_constraints

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
    # print(len(column))
    # print(extra_vars)
    n = len(column) 
    file.write(" ".join([str(x) for x in column]) + " 0\n")
    amo(column, extra_vars, file) 

def write_leakage_clauses_to_file(row,col,extra_vars,indicator_bit, file): 
    n = len(row)
    assert(len(row) == len(col)) 
    product = [row[i] * col[i] for i in range(len(row)) if row[i] * col[i] != 0]
    num_unknown = len([x for x in col if x > 1])
    if indicator_bit: 
        if num_unknown > 0: 
            # assert(len(extra_vars) == len(row))
            file.write(" ".join([str(x) for x in extra_vars]) + " 0\n")
            for i in range(n): 
                file.write(str(-1 * extra_vars[i]) + " " + str(row[i]) + " 0\n")
                file.write(str(-1 * extra_vars[i]) + " " + str(col[i]) + " 0\n")
        else: 
            file.write(" ".join([str(x) for x in product]) + " 0\n")
    else: 
        assert(len(extra_vars) == 0)
        if num_unknown == 0: 
            for i in range(len(product)): 
                file.write(str(-1 * product[i]) + " 0\n")
        else: 
            for i in range(n): 
                file.write(str(-1 * row[i]) + " " + str(-1 * col[i]) + " 0\n")

def parse_output_file(Q_matrix_mapping, D_matrix_mapping, t, r, n, num_vars):
    tmp = {} 
    q_output = [[0 for _ in range(n)] for _ in range(t)]
    d_output = [[0 for _ in range(n)] for _ in range(r)]
    with open('testing.txt') as file: 
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
    for i in range(t): 
        for j in range(n): 
            q_output[i][j] = tmp[Q_matrix_mapping[i][j]]

    for i in range(r): 
        for j in range(n): 
            if D_matrix_mapping[j][i] > 1: 
                d_output[i][j] = tmp[D_matrix_mapping[j][i]]
            else: 
                d_output[i][j] = D_matrix_mapping[j][i]

    return q_output, d_output

class RidSolver:
    def __init__(
        self,
        t_number_of_queries: int,
        r_number_of_records: int,
        n_domain_size: int,
        kdr: int, 
        D_matrix, 
        file_name 
    ):
        self.t_number_of_queries = t_number_of_queries
        self.r_number_of_records = r_number_of_records
        self.n_domain_size = n_domain_size
        self.kdr = kdr
        self.D_matrix = D_matrix
        self.file_name = file_name 
        self.known_records = random.sample([i for i in range(self.r_number_of_records)], int(self.r_number_of_records * self.kdr))
        self.unknown_records = [x for x in range(self.r_number_of_records) if x not in self.known_records]

    def run_iteration(self, leakage): 
        Q_vars, Q_extra_vars, num_Q_vars, num_Q_constraints = compute_Q_info(self.t_number_of_queries, self.n_domain_size)

        D_vars, num_D_vars, num_D_constraints = compute_D_info(self.t_number_of_queries, self.r_number_of_records, self.n_domain_size, self.known_records, self.D_matrix, num_Q_vars)

        rid_extra_vars, num_leakage_vars, num_leakage_constraints = compute_leakage_constraints(self.t_number_of_queries, self.r_number_of_records, self.n_domain_size, leakage, D_vars, num_Q_vars+num_D_vars)
        print(num_Q_constraints, num_D_constraints, num_leakage_constraints)
        f = open(self.file_name, 'w+') 
        # print(str(int(num_Q_vars + num_D_vars + num_leakage_vars)) + " " + str(int(num_Q_constraints + num_D_constraints + num_leakage_constraints)))
        f.write('p cnf ' + str(int(num_Q_vars + num_D_vars + num_leakage_vars)) + " " + str(int(num_Q_constraints + num_D_constraints + num_leakage_constraints)) + "\n") 

        # write the query matrix structure constraints 
        for i in range(self.t_number_of_queries): 
            write_pbeq_clause_to_file(Q_vars[i], Q_extra_vars[i], f)
        
        for i in range(self.r_number_of_records):
            # tmp_vars = np.array(D_vars)[:,i].tolist()
            # count = len([x for x in tmp_vars if x > 1])
            # if count > 0: 
            if i not in self.known_records: 
                f.write(" ".join([str(x) for x in np.array(D_vars)[:,i].tolist()]) + " 0\n")
                # " ".join([str(x) for x in extra_vars]) + " 0\n"
                # write_pbeq_clause_to_file(np.array(D_vars)[:,i].tolist(), D_extra_vars[i], f)

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
        cmd = './build/cadical ' +  self.file_name + ' > testing.txt'
        os.system(cmd) 

        q_output, d_output = parse_output_file(Q_vars, D_vars, self.t_number_of_queries, self.r_number_of_records, self.n_domain_size, num_Q_vars + num_D_vars)

        return q_output, d_output
    
    def solve(self, Q_matrix, D_matrix, leakage):  
        start = time.perf_counter_ns() 
        recovered_Q_matrix, recovered_D_matrix = self.run_iteration(leakage)
        # print(recovered_Q_matrix)

        end = time.perf_counter_ns() 
        runtime = (end - start) / (10 ** 9)

        recovery_rate = sum([recovered_Q_matrix[i] == Q_matrix[i] for i in range(len(Q_matrix))]) / len(Q_matrix)

        # input_D_matrix = [D_matrix[i] for i in self.unknown_records]
        # checked_D_matrix = [recovered_D_matrix[i] for i in self.unknown_records]
        # recovery_rate = compute_recovery_rates(input_D_matrix, checked_D_matrix)
        return runtime, recovery_rate 

def run_one_instance(t_number_of_queries, r_number_of_records, n_domain_size, kdr):
    queries_dist = Zipfian(n_domain_size,2)
    queries = queries_dist.sample(t_number_of_queries)
    #data = Uniform(n_domain_size).sample(r_number_of_records)

    Q_matrix = [[0 for _ in range(n_domain_size)] for _ in range(t_number_of_queries)]
    # D_matrix = [[0 for _ in range(n_domain_size)] for _ in range(r_number_of_records)]

    # with open('allen_p_matrix_200.pkl', 'rb') as f:
    #     D_matrix = pickle.load(f)

    # for i in range(t_number_of_queries): 
    #     Q_matrix[i][queries[i]] = 1
    
    # L = np.matmul(Q_matrix, np.array(D_matrix).T.tolist()) 

    # A_matrix = []
    # for i in range(t_number_of_queries):
    #     A_matrix.append(
    #         queries_dist.pmf() # independent queries 
    #     )

    # rid_solver = RidSolver(
    #     t_number_of_queries=t_number_of_queries,
    #     r_number_of_records=r_number_of_records,
    #     n_domain_size=n_domain_size,
    #     kdr = kdr,
    #     D_matrix = D_matrix,
    #     file_name="test.cnf"
    # )
    recovered_queries = Uniform(n_domain_size).sample(t_number_of_queries)
    recovery_rate = sum([queries[i] == recovered_queries[i] for i in range(len(queries))]) / len(queries)
    time_result = 0 
    # time_result, recovery_rate = rid_solver.solve(Q_matrix, D_matrix, L)
    return time_result, recovery_rate

def main():
    #f = open("fixed_density_t100_n10_r10.txt", "w")
    t_list = [150]
    r_list = [602]
    n_list = [200]
    kdr_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1]
    num_iters = 5
    num_threads = 1

    for t_number_of_queries in t_list: 
        for r_number_of_records in r_list: 
            for n_domain_size in n_list: 
                for kdr in kdr_list: 
                    with Pool(num_threads) as p: 
                        results = p.starmap(run_one_instance, [(t_number_of_queries,r_number_of_records,n_domain_size,kdr,) for _ in range(num_iters)])
                        averages = np.array(results).sum(axis=0) / num_iters
                        print(t_number_of_queries, n_domain_size,kdr, averages[0], averages[1])

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
