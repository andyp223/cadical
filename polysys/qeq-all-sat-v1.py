from typing import List
from z3 import *

from dataclasses import dataclass

import random
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import zipfian, randint

import operator
import functools

from typing import Any

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
    print(t,n)
    Q_vars = [[0 for _ in range(n)] for _ in range(t)]
    for i in range(t): 
        for j in range(n): 
            Q_vars[i][j] = (n * i) + j + 1
    Q_extra_vars = {} 
    extra_vars, extra_constraints = compute_amo_extra_info(n)
    for i in range(t):  
        Q_extra_vars[i] = [t*n + extra_vars * i + j for j in range(1, extra_vars + 1)]

    return Q_vars, Q_extra_vars, t*n + t*extra_vars, t* (extra_constraints + 1)

def compute_num_leakage_constraints(t,n,qeq_leakage, curr):  
    leakage_constraints_extra_vars = {} 
    num_vars = 0
    num_constraints = 0
    for i in range(t): 
        for j in range(i+1,t): 
            if qeq_leakage[i] == qeq_leakage[j]: 
                num_vars += n
                num_constraints += 2*n + 1 
                leakage_constraints_extra_vars[(i,j)] = [curr + j for j in range(1, n + 1)]
                curr += n 
            else: 
                leakage_constraints_extra_vars[(i,j)] = []
                num_constraints += n
    
    return num_vars, num_constraints, leakage_constraints_extra_vars 

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

def parse_output_file(t, n):
    Q_vars = [[0 for _ in range(n)] for _ in range(t)]
    for i in range(t): 
        for j in range(n): 
            Q_vars[i][j] = (n * i) + j + 1
    tmp = {} 
    Q_matrix = [[0 for _ in range(n)] for _ in range(t)]
    Q_boolean = [] 
    with open('testing.txt') as file: 
        while 1: 
            line = next(file).split(" ")
            if line[0] == 's': 
                if line[1] == "UNSATISFIABLE\n": 
                    return -1, -1
                break
        line = next(file).split(" ")
        while line[0] == 'v': 
            for i in range(1, len(line)): 
                # print(int(line[i]))
                tmp[abs(int(line[i]))] = int(int(line[i]) > 0)
                if (int(line[i]) >= t*n): 
                    break
            line = next(file).split(" ")

    for i in range(t): 
        for j in range(n): 
            Q_matrix[i][j] = tmp[Q_vars[i][j]]
            if Q_matrix[i][j] > 0: 
                Q_boolean.append(Q_vars[i][j])
    assert(len(Q_boolean) == t)
    return Q_matrix, Q_boolean 


class QeqSolver:
    def __init__(
        self,
        t_number_of_queries: int,
        n_domain_size: int,
        file_name 
    ):
        self.t_number_of_queries = t_number_of_queries
        self.n_domain_size = n_domain_size
        self.file_name = file_name 
        self.best_sequence = [] 
        self.curr_max = -1 
    
    def compute_score(self,Q_matrix, A_matrix): 
        return sum([Q_matrix[i][j] * A_matrix[i][j] for j in range(self.n_domain_size) for i in range(self.t_number_of_queries)])

    def run_first_iteration(self,qeq_leakage, A_matrix): 
        Q_vars, Q_extra_vars, num_Q_vars, num_Q_constraints = compute_Q_info(self.t_number_of_queries, self.n_domain_size)
        print(Q_vars)
        num_leakage_vars, num_leakage_constraints, leakage_constraints_extra_vars = compute_num_leakage_constraints(self.t_number_of_queries, self.n_domain_size, qeq_leakage, num_Q_vars) 

        f = open(self.file_name, 'w+') 

        f.write('p cnf ' + str(int(num_Q_vars + num_leakage_vars)) + " " + str(int(num_Q_constraints) + int(num_leakage_constraints)) + "\n") 

        # write the query matrix structure constraints 
        for i in range(self.t_number_of_queries): 
            write_pbeq_clause_to_file(Q_vars[i], Q_extra_vars[i], f)

        # write the query leakage constraints  
        for i in range(self.t_number_of_queries):
            for j in range(i+1, self.t_number_of_queries): 
                write_leakage_clauses_to_file(Q_vars[i], Q_vars[j], leakage_constraints_extra_vars[(i,j)], qeq_leakage[i] == qeq_leakage[j], f)
        
        f.close()
        # print(f"Elapsed: {(end - start) / (10 ** 9)} s")
        print("Solving ...")
        start = time.perf_counter_ns() 
        cmd = './build/cadical ' +  self.file_name + ' > testing.txt'
        os.system(cmd) 

        recovered_Q_matrix, recovered_Q_boolean = parse_output_file(self.t_number_of_queries, self.n_domain_size)
        self.best_sequence = recovered_Q_matrix
        self.curr_max = self.compute_score(recovered_Q_matrix, A_matrix) 
        print(self.curr_max) 

        return recovered_Q_matrix, recovered_Q_boolean
    
    def run_next_iteration(self, prev_sequence, A_matrix): 
        if prev_sequence != -1: # there are potentially more solutions 
            with open(self.file_name,'r') as ff:
                data = ff.readlines() 
            
            header = data[0].split(" ")
            header[3] = str(int(header[3]) + 1) + "\n"
            data[0] = " ".join(header)
            data.append(" ".join(str(-1 * x) for x in prev_sequence) + " 0\n")

            with open(self.file_name,'w') as ff: 
                ff.writelines(data)
            
            ff.close()
            cmd = './build/cadical ' +  self.file_name + ' > testing.txt'
            os.system(cmd) 
            recovered_Q_matrix, recovered_Q_boolean = parse_output_file(self.t_number_of_queries, self.n_domain_size)
            
            return recovered_Q_matrix, recovered_Q_boolean 

    def solve(self, actual_Q, qeq_leakage, A_matrix, num_iters):  
        start = time.perf_counter_ns() 
        recovered_Q_matrix, recovered_Q_boolean = self.run_first_iteration(qeq_leakage, A_matrix)

        for i in range(num_iters): 
            print(i)
            recovered_Q_matrix, recovered_Q_boolean = self.run_next_iteration(recovered_Q_boolean, A_matrix) 
            if recovered_Q_boolean != -1: 
                if self.compute_score(recovered_Q_matrix, A_matrix) > self.curr_max: 
                    print("ENTERS")
                    self.curr_max = self.compute_score(recovered_Q_matrix, A_matrix)
                    self.best_sequence = recovered_Q_matrix
            else:
                break 

        end = time.perf_counter_ns() 
        runtime = (end - start) / (10 ** 9)
        recovered_Q = [] 
        for i in range(self.t_number_of_queries): 
            for j in range(self.n_domain_size): 
                if self.best_sequence[i][j] > 0: 
                    recovered_Q.append(j) 
        assert(len(recovered_Q) == self.t_number_of_queries)
        print(actual_Q)
        print(recovered_Q)
        recovery_rate = sum([x == y for x,y in zip(recovered_Q, actual_Q)])/self.t_number_of_queries
        return runtime, recovery_rate 

def run_one_instance(t_number_of_queries, n_domain_size):
    distribution = Zipfian(n_domain_size, 2)
    actual_Q = distribution.sample(t_number_of_queries)

    random_F = [i for i in range(n_domain_size)]
    random.shuffle(random_F)

    qeq_leakage = [random_F[q] for q in actual_Q]

    A_matrix = []
    for i in range(t_number_of_queries):
        A_matrix.append(
            distribution.pmf() # independent queries 
        )

    qeqsolver = QeqSolver(
        t_number_of_queries=t_number_of_queries,
        n_domain_size=n_domain_size,
        file_name="test.cnf"
    )

    time_result, recovery_rate = qeqsolver.solve(actual_Q, qeq_leakage, A_matrix, 1000)
    return time_result, recovery_rate

def main():
    #f = open("fixed_density_t100_n10_r10.txt", "w")
    t_list = [500]
    n_list = [20]
    num_iters = 1
    num_threads = 1

    for t_number_of_queries in t_list: 
        for n_domain_size in n_list: 
                with Pool(num_threads) as p: 
                    results = p.starmap(run_one_instance, [(t_number_of_queries,n_domain_size,) for _ in range(num_iters)])
                    averages = np.array(results).sum(axis=0) / num_iters
                    print(t_number_of_queries, n_domain_size,averages[0], averages[1])

if __name__ == "__main__":
    main()
