from typing import List
from z3 import *

from dataclasses import dataclass
from multiprocessing import Pool

import random
import time
import math

import numpy as np
from scipy.stats import zipfian, randint

import operator
import functools

from typing import Any
import os 

def parse_output_file(file_name,D_matrix_mapping, n, r, num_vars):
    tmp = {}
    output = [[0 for _ in range(n)] for _ in range(r)] 
    with open('testing.txt') as file: 
        while 1: 
            line = next(file).split(" ")
            if line[0] == 's': 
                break
        line = next(file).split(" ")
        while line[0] == 'v': 
            for i in range(1, len(line)): 
                tmp[abs(int(line[i]))] = int(int(line[i]) > 0)
                if (int(line[i]) >= num_vars): 
                    break
                # tmp.append(int(line[i]))
            line = next(file).split(" ")
    
    for i in range(r): 
        for j in range(n): 
            output[i][j] = tmp[D_matrix_mapping[(j,i)]]

    return output

def write_clauses_to_file(clauses, file): 
    for clause in clauses: 
        file.write(" ".join(clause) + " 0\n")

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

# def write_range_clauses_to_file(clauses,extra_vars,file): 
#     n = len(clauses)
#     file.write(" ".join([str(x) for x in extra_vars]) + " 0\n")
#     for i in range(n): 
#         clause = clauses[i]
#         for j in range(len(clause)): 
#             file.write(str(-1 * extra_vars[i]) + " " + str(clauses[i][j]) + " 0\n")

def compute_D_extra_info(n): 
    if n <= 4: 
        return 0, int(n * (n - 1) / 2)
    else: 
        extra_vars, extra_constraints = compute_D_extra_info(n - 2)
        return 1 + extra_vars, 6 + extra_constraints

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

def write_pbeq_clause(column, extra_vars, file):
    n = len(column) 
    # clauses.append([str(x) for x in column]) 
    file.write(" ".join([str(x) for x in column]) + " 0\n")
    amo(column, extra_vars, file) 
    # for i in range(n):
    #     for j in range(i + 1,n): 
    #         clauses.append([str(-1 * column[i]),str(-1 * column[j])])

    # write_clauses_to_file(clauses,file) 

def write_R_clauses_to_file(clauses, extra_vars, file):
    assert(len(clauses) == len(extra_vars))
    n = len(clauses) 
    file.write(" ".join([str(x) for x in extra_vars]) + " 0\n")
    for i in range(n): 
        file.write(str(-1 * extra_vars[i]) + " " + str(clauses[i][0]) + " 0\n")
        file.write(str(-1 * extra_vars[i]) + " " + str(clauses[i][1]) + " 0\n")

def NPArray(n, prefix=None, dtype=IntSort()):
    return np.array([FreshConst(dtype, prefix=prefix) for i in range(n)])

# check whether array contains all numbers between 0 and n - 1
def compute_density(A, n): 
    B=np.arange(n)
    mask = np.ones(len(B), dtype=bool)
    mask[A] = False
    out = B[mask]
    return 1 - (len(out)/float(n)), len(out) == 0

class Distribution:
    def sample(self, number_of_queries: int) -> List[int]: ...

class Uniform(Distribution):
    def __init__(self, domain_size: int):
        self.domain_size = domain_size

    def sample(self, number_of_queries: int) -> List[int]:
        return randint.rvs(0, self.domain_size, size=number_of_queries)

class Zipfian(Distribution):
    def __init__(self, domain_size: int, s: int):
        self.domain_size = domain_size
        self.s = s

    def sample(self, number_of_queries: int) -> List[int]:
        return zipfian.rvs(self.s, self.domain_size, size=number_of_queries, loc=-1)

class FixedDensity(Distribution):
    def __init__(self, domain_size: int, density):
        self.domain_size = domain_size
        self.density = density

    def sample(self, number_of_queries: int) -> List[int]:
        fixed_domain_size = int(self.density * self.domain_size)
        assert(number_of_queries >= fixed_domain_size) 

        possible_outputs = random.sample(range(self.domain_size), fixed_domain_size)
        samples = randint.rvs(0, fixed_domain_size, size=number_of_queries - fixed_domain_size)
        output = np.array(possible_outputs + [possible_outputs[i] for i in samples])
        random.shuffle(output) 
        return output

def Abs(x):
    """
    Utility function to compute absolute value of a Z3 expression.
    """
    return If(x >= 0, x, -x)

class RangeRidSolver:
    def __init__(
        self,
        t_number_of_queries: int,
        n_domain_size: int,
        r_number_of_records: int,
        file_name 
    ):
        self.t_number_of_queries = t_number_of_queries
        self.n_domain_size = n_domain_size
        self.r_number_of_records = r_number_of_records
        self.set_of_ranges = []
        self.file_name = file_name
        for i in range(n_domain_size):
            for j in range(i, n_domain_size):
                    self.set_of_ranges.append((i,j))

    def solve(
        self,
        R_matrix
    ):
        assert(len(R_matrix) == self.t_number_of_queries)

        R_num_ones = sum([sum(x) for x in R_matrix])
        amo_extra_vars, amo_constraints = compute_D_extra_info(self.n_domain_size)

        num_Q_vars = self.t_number_of_queries * self.n_domain_size
        num_D_vars = self.n_domain_size * self.r_number_of_records
        num_R_vars = 0

        num_Q_extra_vars = self.t_number_of_queries * (2 * self.n_domain_size) 
        num_D_extra_vars = self.r_number_of_records * amo_extra_vars
        num_R_extra_vars = R_num_ones * self.n_domain_size 

        num_Q_constraints_prev = self.t_number_of_queries * (1 + len(self.set_of_ranges)* self.n_domain_size) 
        num_Q_constraints = self.t_number_of_queries * (5 * self.n_domain_size - 1) 
        num_D_constraints_prev = int(self.r_number_of_records * (self.n_domain_size * (self.n_domain_size - 1)/2 + 1))
        num_D_constraints = self.r_number_of_records * (amo_constraints + 1)
        num_R_constraints = R_num_ones * (2 * self.n_domain_size + 1) + (self.t_number_of_queries * self.r_number_of_records - R_num_ones) * self.n_domain_size

        num_vars = num_Q_vars + num_D_vars + num_R_vars
        num_extra_vars = num_Q_extra_vars + num_D_extra_vars + num_R_extra_vars 
        num_constraints = num_Q_constraints + num_D_constraints + num_R_constraints
        num_total_vars_prev = num_vars + self.t_number_of_queries * len(self.set_of_ranges)
        num_total_vars = num_vars + num_extra_vars

        print(num_total_vars, num_constraints) 

        # Construct query matrix.
        #print("Constructing Q...")
        f = open(self.file_name, "w+")

        f.write('p cnf' + " " + str(num_total_vars) + " " + str(num_constraints) + " \n")
        # f.write('p cnf' + " " + str(num_total_vars_prev) + " " + str(num_Q_constraints_prev + num_D_constraints + num_R_constraints) + " \n")

        print("Constructing Q...")
        start = time.perf_counter_ns()
        Q_matrix =  {} 
        for i in range(self.t_number_of_queries): 
            for j in range(self.n_domain_size): 
                Q_matrix[(i,j)] = i * self.n_domain_size + j + 1

        Q_extra_vars = {} 
        for i in range(self.t_number_of_queries): 
            Q_extra_vars[i] = {} 
            Q_extra_vars[i][0] = [num_vars + (2*i) * self.n_domain_size + j for j in range(1,self.n_domain_size + 1) ] # after variables
            Q_extra_vars[i][1] = [num_vars + (2*i + 1) * self.n_domain_size + j for j in range(1,self.n_domain_size + 1) ] # before variables

            # Q_extra_vars[i] = [num_vars + i * len(self.set_of_ranges) + j for j in range(1,len(self.set_of_ranges) + 1)]
        
        for i in range(self.t_number_of_queries): 
            Q_vars = []
            for j in range(self.n_domain_size): 
                Q_vars.append(Q_matrix[(i,j)])
            write_range_clauses_to_file(Q_vars,Q_extra_vars[i],f)

        # for i in range(self.t_number_of_queries):
        #     clause=[]
        #     for j in range(len(self.set_of_ranges)):
        #         (lb,ub) = self.set_of_ranges[j]
        #         components = []
        #         for k in range(self.n_domain_size):
        #             if (k<lb):
        #                 components.append(-1* Q_matrix[(i,k)])
        #             elif (k>ub):     
        #                 components.append(-1 * Q_matrix[(i,k)])
        #             else:
        #                 components.append(Q_matrix[(i,k)])
        #         #  print(components)                          
        #         clause.append(components)
        #     write_range_clauses_to_file(clause, Q_extra_vars[i], f)

        # Construct data matrix.
        end = time.perf_counter_ns()
        
        print(f"Elapsed: {(end - start) / (10 ** 9)} s")
        print("Constructing D...")
        start = time.perf_counter_ns()
        D_matrix = {}
        for i in range(self.n_domain_size): 
            for j in range(self.r_number_of_records): 
                D_matrix[(i,j)] = (self.t_number_of_queries * self.n_domain_size) + i*self.r_number_of_records + j + 1
        
        D_extra_vars = {} 
        for i in range(self.r_number_of_records):
            D_extra_vars[i] = [(num_vars + num_Q_extra_vars + i * amo_extra_vars + j) for j in range(1, amo_extra_vars + 1)]

        for i in range(self.r_number_of_records):
            column = []
            for j in range(self.n_domain_size):
                column.append(D_matrix[(j,i)])
            
            # exactly one of these are true 
            write_pbeq_clause(column, D_extra_vars[i], f)
        
        R_matrix_extra_vars = {} 
        num_prev_vars = num_vars + num_Q_extra_vars + num_D_extra_vars
        for i in range(R_num_ones): 
            R_matrix_extra_vars[i] =[(num_prev_vars + self.n_domain_size* i + j) for j in range(1,self.n_domain_size + 1)]

        end = time.perf_counter_ns()
        
        print(f"Elapsed: {(end - start) / (10 ** 9)} s")

        print("Constructing Q * D = R constraint...")
        start = time.perf_counter_ns()
        curr_count = 0
        for i in range(self.t_number_of_queries):
            for j in range(self.r_number_of_records):
                if R_matrix[i][j]:
                    clauses = []
                    for k in range(self.n_domain_size):
                        clauses.append([Q_matrix[(i,k)], D_matrix[(k,j)]])
                        # clause.append(And(Q_matrix[i][k],D_matrix[k][j]))
                    write_R_clauses_to_file(clauses,R_matrix_extra_vars[curr_count],f) 
                    # solver.add(Or(clause))
                    curr_count += 1
                else:     
                    clauses = []
                    for k in range(self.n_domain_size):
                        clauses.append([str(-1 * Q_matrix[(i,k)]), str(-1 * D_matrix[(k,j)])])
                        # clause.append(Or(Not(Q_matrix[i][k]),Not(D_matrix[k][j])))
                    write_clauses_to_file(clauses,f)
 
        f.close()
        end = time.perf_counter_ns()
        
        print(f"Elapsed: {(end - start) / (10 ** 9)} s")
        print("Solving ...")
        start = time.perf_counter_ns() 
        cmd = './build/cadical ' +  self.file_name + ' > testing.txt'
        os.system(cmd) 

        recovered_D_matrix = parse_output_file('testing.txt', D_matrix, self.n_domain_size, self.r_number_of_records, num_vars)

        #print("Optimizing objective...")
        # start = time.perf_counter_ns()

        # elapsed = time.perf_counter_ns() - start
        #print(f"Elapsed: {elapsed / (10 ** 9)} s")

        # print(f"Recovered queries: {actual_Q_matrix}")

        # recovered_D_matrix = [[0 for _ in range(self.n_domain_size)] for _ in range(self.r_number_of_records)] 
        # for i in range(self.r_number_of_records): 
        #     for j in range(self.n_domain_size): 
        #         recovered_D_matrix[i][j] = model[D_matrix[j][i]].__bool__()
        # print(recovered_D_matrix)

        end = time.perf_counter_ns()
        
        print(f"Elapsed: {(end - start) / (10 ** 9)} s")
        return recovered_D_matrix

def compute_mae(data,recovered,domain_size): 
    errors = np.absolute([float(recovered[x] - data[x])/domain_size for x in range(len(data))])
    return float(sum(errors)) / len(errors)

def compute_recovery_rates(matrix, recovered_matrix): 
    r_number_of_records = len(matrix)
    n_domain_size = len(matrix[0]) 

    matrix_idx = [np.nonzero(x)[0][0] for x in matrix]
    recovered_matrix_idx = [np.nonzero(x)[0][0] for x in recovered_matrix]
    recovered_matrix_idx_reversed = [(n_domain_size - 1) - x for x in recovered_matrix_idx]

    print(matrix_idx)
    print(recovered_matrix_idx)

    num_wrong_left = np.count_nonzero(np.sum(np.abs(recovered_matrix - matrix),axis=1))
    num_wrong_right = np.count_nonzero(np.sum(np.abs(np.flip(recovered_matrix,axis=1) - matrix),axis=1))

    recovery_rate = (r_number_of_records - min(num_wrong_left,num_wrong_right))/float(r_number_of_records)

    # Compute approximation metric here
    approx_metric_left = sum([abs(i - j) for i,j in zip(matrix_idx, recovered_matrix_idx)])
    approx_metric_right = sum([abs(i - j) for i,j in zip(matrix_idx, recovered_matrix_idx_reversed)])

    approx_recovery_rate = 1 - min(approx_metric_left, approx_metric_right)/(math.floor(n_domain_size * r_number_of_records))

    return recovery_rate, approx_recovery_rate

def run_one_instance(t_number_of_queries, n_domain_size, r_number_of_records, data_dist, query_dist, density_pct):

    # number of possible range queries
    number_of_ranges = n_domain_size *(n_domain_size + 1) /2

    # Step 1: generate the list of all possible ranges
    set_of_ranges = []
    for i in range(n_domain_size):
        for j in range(i, n_domain_size):
                set_of_ranges.append((i,j))
    number_of_ranges = len(set_of_ranges)

    # print(set_of_ranges) 
    # Generating a random Q matrix 
    if query_dist == "zipf": 
        distribution = Zipfian(number_of_ranges, 2) # TODO: Add this to parameter 
    elif query_dist == "uniform": 
        distribution = Uniform(number_of_ranges)
    else:
        print(query_dist)

    samples = distribution.sample(t_number_of_queries) 

    Q_matrix = [[0 for _ in range(n_domain_size)] for _ in range(t_number_of_queries)]
    for i in range(t_number_of_queries): 
        (lb,ub) = set_of_ranges[samples[i]]
        for j in range(lb,ub+1):
            Q_matrix[i][j] = 1

    # Generating a random D matrix (using uniform distribution)
    D_matrix = [[0 for _ in range(n_domain_size)] for _ in range(r_number_of_records)] 

    if data_dist == "fixed_density": 
        uniform_samples = FixedDensity(n_domain_size,density_pct).sample(r_number_of_records)
    elif data_dist == "uniform": 
        uniform_samples = Uniform(n_domain_size).sample(r_number_of_records)

    for i in range(r_number_of_records): 
        D_matrix[i][uniform_samples[i]] = 1

    # Compute the leakage R_matrix from Q_matrix and D_matrix 
    R_matrix = [[0 for _ in range(r_number_of_records)] for _ in range(t_number_of_queries)]

    for i in range(t_number_of_queries):
        for j in range(r_number_of_records):
             for k in range(n_domain_size):
                  R_matrix[i][j] = R_matrix[i][j] or (Q_matrix[i][k] and D_matrix[j][k]) 

    rangeridsolver = RangeRidSolver(
        t_number_of_queries=t_number_of_queries,
        n_domain_size=n_domain_size,
        r_number_of_records=r_number_of_records,
        file_name = "test.cnf"
    )

    recovered_D_matrix = rangeridsolver.solve(R_matrix=R_matrix)
    D_matrix = np.array(D_matrix)
    recovered_D_matrix = np.array(recovered_D_matrix)

    # Generating a random D matrix (using uniform distribution)
    uniform_guess_matrix = [[0 for _ in range(n_domain_size)] for _ in range(r_number_of_records)] 
    guess_samples = Uniform(n_domain_size).sample(r_number_of_records)
    #  print(uniform_samples)

    for i in range(r_number_of_records): 
        uniform_guess_matrix[i][guess_samples[i]] = 1
    recovery, approx_recovery = compute_recovery_rates(D_matrix, recovered_D_matrix)
    random_recovery, random_approx_recovery = compute_recovery_rates(D_matrix,uniform_guess_matrix) 

    density, is_dense = compute_density(uniform_samples,n_domain_size)

    return recovery, approx_recovery, random_recovery, random_approx_recovery, density, is_dense

def main():
    f = open("fixed_density_t200_n10_r10.txt", "w")
    t_list = [16]
    n_list = [16]
    r_list = [16]
    data_dist = "fixed_density" #uniform or fixed_density for now
    query_dist = "uniform" #uniform or zipf
    num_iters = 10
    num_threads = 1

    for t_number_of_queries in t_list: 
        for n_domain_size in n_list: 
            for r_number_of_records in r_list: 
                print(t_number_of_queries, n_domain_size, r_number_of_records)

                with Pool(num_threads) as p: 
                    if data_dist == "fixed_density": 
                        densities = [1]
                    else: 
                        densities = [1]
                    for density_pct in densities:
                        start = time.time()

                        results = p.starmap(run_one_instance, [(t_number_of_queries,n_domain_size,r_number_of_records, data_dist, query_dist, density_pct,) for _ in range(num_iters)])
                        averages = np.array(results).sum(axis=0) / num_iters
                
                        end = time.time()
                
                        print(averages)
                        f.write(str(t_number_of_queries) + ", " + str(r_number_of_records) + ", " + ", ".join(str(round(x, 2)) for x in averages))
                        f.write("\n")
    f.close()


if __name__ == "__main__":
    start = time.perf_counter_ns()
    main()
    # print(compute_D_extra_info(200))
    # run_one_instance(100,100,100,"fixed_density",'uniform',1)
    
    print(compute_mae([49,28,50,55,75,95,26,71,19,99],[97,97,97,97,97,95,98,97,99,95],100))
    print(compute_mae([3,4,80,34,67,25,17,92,5,77],[8,9,100,100,100,100,4,100,9,100],100))
    end = time.perf_counter_ns()
    print(f"Elapsed: {(end - start) / (10 ** 9)} s")