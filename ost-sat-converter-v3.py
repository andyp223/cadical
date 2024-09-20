from typing import List
from z3 import *

from dataclasses import dataclass
from multiprocessing import Pool
from collections import defaultdict

import random
import time
import math

import numpy as np
from scipy.stats import zipfian, randint

import operator
import functools

from typing import Any
import os 
import sys 

sys.setrecursionlimit(5000)

def gen_set_of_edges(n_domain_size): 
    output = {}  
    length = 1
    left = 0
    count = 0
    while length <= n_domain_size: 
        right = left + length - 1
        if right < n_domain_size: 
            output[count] = (left, right)
            left += length
            count += 1
        else: 
            left = 0
            length *= 2
    return output 

def gen_range_min_cover(range_query, n_domain_size): 
    output = [0 for _ in range(2*n_domain_size - 1)]

    for i in range(range_query[0],range_query[1] + 1): 
        output[i] = 1
    prev_level_start_index = 0 
    level_size = n_domain_size // 2

    while level_size > 0: 
        for i in range(level_size): 
            left_node = prev_level_start_index + 2*i 
            right_node = prev_level_start_index + 2*i + 1 

            if output[left_node] & output[right_node]: 
                output[prev_level_start_index + 2*level_size + i] = 1
                output[left_node] = 0
                output[right_node] = 0
            
        prev_level_start_index += 2*level_size 
        level_size //= 2
    return output

def gen_range_to_min_cover(set_of_ranges, n_domain_size): 
    output = {}
    for i in range(len(set_of_ranges)): 
        r = set_of_ranges[i]
        output[i] = gen_range_min_cover(r, n_domain_size)

    return output  

def compute_hypergraph_info(n_domain_size, set_of_ranges):
    s = 2*n_domain_size - 1 
    set_of_edges = gen_set_of_edges(n_domain_size) 
    range_to_min_cover = gen_range_to_min_cover(set_of_ranges, n_domain_size) 

    m = max([sum(x) for _,x in range_to_min_cover.items()])

    E_matrix = [[0 for _ in range(s)] for _ in range(n_domain_size)]
    list_of_H_matrix = [[[0 for _ in range(s)] for _ in range(len(set_of_ranges))] for _ in range(m)]

    # making E_matrix 
    for i,edge in set_of_edges.items(): 
        (left, right) = edge
        for j in range(left, right + 1):
            E_matrix[j][i] = 1

    # making H matrices
    for i,range_min_cover in range_to_min_cover.items(): 
        count = 0
        for j in range(len(range_min_cover)): 
            if range_min_cover[j] == 1: 
                list_of_H_matrix[count][i][j] = 1
                count += 1 

    return set_of_edges, range_to_min_cover, E_matrix, list_of_H_matrix

def gen_qeq_leakage(queries, range_to_min_cover): 
    output = {}
    t = len(queries)
    for i in range(t): 
        for j in range(i + 1,t): 
            r1 = np.array(range_to_min_cover[queries[i]]).nonzero()[0]
            r2 = np.array(range_to_min_cover[queries[j]]).nonzero()[0]
            A =  [[0 for _ in range(len(r2))] for _ in range(len(r1))]
            for k in range(len(r1)): 
                for l in range(len(r2)): 
                    if r1[k] == r2[l]: 
                        A[k][l] = 1
            output[(i,j)] = A
    return output 

def gen_rid_leakage(queries, data, set_of_edges, range_to_min_cover):
    output = []
    for query in queries: 
        r = np.array(range_to_min_cover[query]).nonzero()[0]
        B = [[0 for _ in range(len(data))] for _ in range(len(r))]
        for i in range(len(r)): 
            edge = r[i] 
            (left, right) = set_of_edges[edge]
            for j in range(len(data)): 
                if left <= data[j] and data[j] <= right: 
                    B[i][j] = 1
        output.append(B)
    return output

def gen_S_sets(H_matrices, range_to_min_cover): 
    output = {}
    num_ranges = len(H_matrices[0])
    for i in range(num_ranges): 
        for j in range(i, num_ranges): 
            alpha = sum(range_to_min_cover[i])
            beta = sum(range_to_min_cover[j])
            lst1 = np.nonzero(range_to_min_cover[i])[0]
            lst2 = np.nonzero(range_to_min_cover[j])[0]

            matrix = [[lst1[i] == lst2[j] for j in range(beta)] for i in range(alpha)] 
            matrix_t = list(map(list, zip(*matrix)))

            if (alpha,beta) in output: 
                output[(alpha,beta)][tuple(map(tuple, matrix))].append((i,j))
            else: 
                output[(alpha,beta)] = defaultdict(list)
                output[(alpha,beta)][tuple(map(tuple, matrix))].append((i,j))
            
            if i != j: 
                if (beta,alpha) in output: 
                    output[(beta,alpha)][tuple(map(tuple, matrix_t))].append((j,i))
                else: 
                    output[(beta,alpha)] = defaultdict(list)
                    output[(beta,alpha)][tuple(map(tuple, matrix_t))].append((j,i))
    
    return output 

def gen_T_sets(H_matrices, E_matrix): 
    output = {} 
    n = len(H_matrices) 
    num_ranges = len(H_matrices[0])
    N = len(E_matrix) 

    for alpha in range(num_ranges): 
        edge_mapping = {} 
        for j in range(n): 
            tmp = []
            for x in range(N): 
                row1 = H_matrices[j][alpha]
                row2 = E_matrix[x]
                if sum([a*b for a,b in zip(row1,row2)]) == 1: 
                    tmp.append(x) 
            if len(tmp) > 0: 
                edge_mapping[j] = tmp 
        output[alpha] = edge_mapping 
    
    return output 

def compute_qeq_extra_vars(S, qeq_leakage, t, curr_num): 
    extra_vars = 0
    extra_constraints = 0
    output = {}
    for i in range(t): 
        for j in range(i+1,t): 
            A = qeq_leakage[(i,j)]
            a = len(A) 
            b = len(A[0])
            
            output[(i,j)] = [curr_num + 1 + x for x in range(len(S[(a,b)][tuple(map(tuple,A))]))]
            extra_vars += len(S[(a,b)][tuple(map(tuple,A))])
            extra_constraints += 2 * len(S[(a,b)][tuple(map(tuple,A))]) + 1
            curr_num += len(S[(a,b)][tuple(map(tuple,A))])

    return extra_vars, extra_constraints, output 

def compute_rid_extra_vars(T, query_sequence, rid_leakage, t, curr_num): 
    extra_vars = 0
    extra_constraints = 0
    output = {}
    for i in range(t): 
        B = rid_leakage[i] 
        query_num = query_sequence[i]

        for j in range(len(B)): 
            for k in range(len(B[0])): 
                if B[j][k]: 
                    output[(i,j,k)] = [curr_num + 1 + x for x in range(len(T[query_num][j]))]
                    # xtra_vars += len(T[query_num][j])
                    # extra_vars += 0 
                    # extra_constraints += 2 * len(T[j]) + 1
                    extra_constraints += 1 
                    # curr_num += len(T[j]) 

                else: 
                    output[(i,j,k)] = []
                    extra_constraints += len(T[query_num][j])
    # print("hello")
    # print(extra_vars, extra_constraints)
    return extra_vars, extra_constraints, output 

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

def parse_output_file(file_name,D_matrix_mapping, n, r, num_vars):
    tmp = {}
    output = [[0 for _ in range(n)] for _ in range(r)] 
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
                tmp[abs(int(line[i]))] = int(int(line[i]) > 0)
                if (int(line[i]) >= num_vars): 
                    break
                # tmp.append(int(line[i]))
            line = next(file).split(" ")
    
    for i in range(r): 
        for j in range(n): 
            output[i][j] = tmp[D_matrix_mapping[j][i]]

    return output

def parse_range_output_file(file_name,t, num_ranges, num_vars):
    tmp = {}
    boolean_output = []
    query_output= []
    with open(file_name) as file: 
        while 1: 
            line = next(file).split(" ")
            if line[0] == 's': 
                if line[1] == "UNSATISFIABLE\n":
                    return -1, -1
                break
        line = next(file).split(" ")
        while line[0] == 'v': 
            for i in range(1, len(line)): 
                if (int(line[i]) > num_vars): 
                    break
                if int(int(line[i])) > 0: 
                    boolean_output.append(abs(int(line[i])))
                    query_output.append((abs(int(line[i])) - 1) % num_ranges)
            line = next(file).split(" ")
    # print(query_output)
    # print(boolean_output)
    assert(len(boolean_output) == t) 
    return boolean_output, query_output

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

def compute_amo_extra_info(n): 
    if n <= 4: 
        return 0, int(n * (n - 1) / 2)
    else: 
        extra_vars, extra_constraints = compute_amo_extra_info(n - 2)
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

class OSTLeakageSolver:
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

        self.set_of_edges,self.range_to_min_cover,self.E_matrix, self.list_of_H_matrix = compute_hypergraph_info(self.n_domain_size,self.set_of_ranges)
    
    def compute_all_possible_range_matrices(self,qeq_leakage): 
    
        # Preparing all the boolean variables for cadical 

        amo_R_extra_vars, amo_R_extra_constraints = compute_amo_extra_info(len(self.set_of_ranges))
        amo_D_extra_vars, amo_D_extra_constraints = compute_amo_extra_info(self.n_domain_size)

        num_R_vars = self.t_number_of_queries * len(self.set_of_ranges)
        # num_D_vars = self.n_domain_size * self.r_number_of_records

        # num_vars = num_R_vars + num_D_vars
        num_vars = num_R_vars 

        num_R_extra_vars = self.t_number_of_queries * amo_R_extra_vars
        # um_D_extra_vars = self.r_number_of_records * amo_D_extra_vars

        R_matrix =  [[0 for _ in range(len(self.set_of_ranges))] for _ in range(self.t_number_of_queries)]
        for i in range(self.t_number_of_queries): 
            for j in range(len(self.set_of_ranges)): 
                R_matrix[i][j]= i * len(self.set_of_ranges) + j + 1
        
        R_extra_vars = {} 
        for i in range(self.t_number_of_queries):
            R_extra_vars[i] = [(num_vars + i * amo_R_extra_vars + j) for j in range(1, amo_R_extra_vars + 1)]

        print("gen s sets")
        start = time.perf_counter_ns()
        S = gen_S_sets(self.list_of_H_matrix, self.range_to_min_cover)
        end = time.perf_counter_ns()
        print(f"Gen S Sets: {(end - start) / (10 ** 9)} s")

        num_qeq_extra_vars, num_qeq_constraints, qeq_extra_vars = compute_qeq_extra_vars(S, qeq_leakage, self.t_number_of_queries, num_vars + num_R_extra_vars)
        # num_rid_extra_vars, num_rid_constraints, rid_extra_vars = compute_rid_extra_vars(T, rid_leakage, self.t_number_of_queries, num_vars + num_R_extra_vars + num_D_extra_vars + num_qeq_extra_vars)
        
        num_R_constraints = self.t_number_of_queries * (amo_R_extra_constraints + 1)
        # num_D_constraints = self.r_number_of_records * (amo_D_extra_constraints + 1)

        # num_extra_vars = num_R_extra_vars + num_D_extra_vars + num_qeq_extra_vars + num_rid_extra_vars 
        num_extra_vars = num_R_extra_vars + num_qeq_extra_vars 
        num_total_vars = num_vars + num_extra_vars 

        # num_total_constraints = num_R_constraints + num_D_constraints + num_qeq_constraints + num_rid_constraints
        num_total_constraints = num_R_constraints + num_qeq_constraints

        # num_total_vars = num_vars + num_R_extra_vars + num_D_extra_vars + num_qeq_extra_vars
        # num_total_constraints = num_R_constraints + num_D_constraints + num_qeq_constraints

        # print(num_R_vars, num_D_vars, num_R_extra_vars, num_D_extra_vars, num_qeq_extra_vars, num_rid_extra_vars)
        print("num constraints")
        # print(num_R_constraints, num_qeq_constraints)
        # print(num_R_constraints, num_D_constraints, num_qeq_constraints, num_rid_constraints) 

        # Construct query matrix.
        #print("Constructing Q...")
        f = open(self.file_name, "w+")

        f.write('p cnf' + " " + str(num_total_vars) + " " + str(num_total_constraints) + " \n")
        # f.write('p cnf' + " " + str(num_total_vars_prev) + " " + str(num_Q_constraints_prev + num_D_constraints + num_R_constraints) + " \n")

        print("Constructing Q...")
        start = time.perf_counter_ns()
        
        for i in range(self.t_number_of_queries): 
            R_vars = []
            for j in range(len(self.set_of_ranges)): 
                R_vars.append(R_matrix[i][j])
            write_pbeq_clause(R_vars,R_extra_vars[i],f)

        end = time.perf_counter_ns()
        
        print(f"Elapsed: {(end - start) / (10 ** 9)} s")

        print("Constructing QEQ Leakage constraint...")
        # print(qeq_leakage)
        start = time.perf_counter_ns()
        for i in range(self.t_number_of_queries):
            for j in range(i + 1,self.t_number_of_queries):
                A = qeq_leakage[(i,j)]
                a = len(A) 
                b = len(A[0])
                
                matches = S[(a,b)][tuple(map(tuple,A))]
                row = []
                col = [] 

                for match in matches: 
                    row.append(R_matrix[i][match[0]])
                    col.append(R_matrix[j][match[1]])
                
                write_leakage_clauses_to_file(row,col,qeq_extra_vars[(i,j)],1,f)
 
        f.close()
        end = time.perf_counter_ns()
        
        print(f"Elapsed: {(end - start) / (10 ** 9)} s")

        output = [] 

        print("Solving ...")
        start = time.perf_counter_ns() 
        cmd = './build/cadical ' +  self.file_name + ' > testing.txt'
        os.system(cmd) 
        boolean_output, tmp = parse_range_output_file('testing.txt', self.t_number_of_queries, len(self.set_of_ranges), num_R_vars)

        while tmp != -1: # there are potentially more solutions 
            output.append(boolean_output) 

            with open('test.cnf','r') as ff:
                data = ff.readlines() 
            
            header = data[0].split(" ")
            header[3] = str(int(header[3]) + 1)
            data[0] = " ".join(header)
            data.append(" ".join(str(-1 * x) for x in boolean_output) + " 0\n")

            with open('test.cnf','w') as ff: 
                ff.writelines(data)
            
            os.system(cmd) 
            boolean_output, tmp = parse_range_output_file('testing.txt', self.t_number_of_queries, len(self.set_of_ranges), num_R_vars)
            print(tmp)

        end = time.perf_counter_ns()
        print(f"Elapsed: {(end - start) / (10 ** 9)} s")
        print(len(output))
        return output


    def solve(self,boolean_outputs,rid_leakage):

        # Preparing all the boolean variables for cadical 

        # amo_R_extra_vars, amo_R_extra_constraints = compute_amo_extra_info(len(self.set_of_ranges))
        amo_D_extra_vars, amo_D_extra_constraints = compute_amo_extra_info(self.n_domain_size)

        # num_R_vars = self.t_number_of_queries * len(self.set_of_ranges)
        num_D_vars = self.n_domain_size * self.r_number_of_records

        # num_vars = num_R_vars + num_D_vars
        num_vars = num_D_vars 

        # num_R_extra_vars = self.t_number_of_queries * amo_R_extra_vars
        num_D_extra_vars = self.r_number_of_records * amo_D_extra_vars

        D_extra_vars = {} 
        for i in range(self.r_number_of_records):
            D_extra_vars[i] = [(num_vars + i * amo_D_extra_vars + j) for j in range(1, amo_D_extra_vars + 1)]

        print("gen t sets")
        start = time.perf_counter_ns()
        T = gen_T_sets(self.list_of_H_matrix, self.E_matrix)
        end = time.perf_counter_ns()
        print(f"Gen T Sets: {(end - start) / (10 ** 9)} s")

        
        for boolean_output in boolean_outputs: 
            query_sequence = [(x - 1) % len(self.set_of_ranges) for x in boolean_output]

            print("GUESSED SEQUENCE")
            print(query_sequence) 
            # num_qeq_extra_vars, num_qeq_constraints, qeq_extra_vars = compute_qeq_extra_vars(S, qeq_leakage, self.t_number_of_queries, num_vars + num_R_extra_vars)
            
            num_rid_extra_vars, num_rid_constraints, rid_extra_vars = compute_rid_extra_vars(T, query_sequence, rid_leakage, self.t_number_of_queries, num_vars + num_D_extra_vars)
            # print(num_rid_extra_vars, num_rid_constraints, rid_extra_vars)

            # num_R_constraints = self.t_number_of_queries * (amo_R_extra_constraints + 1)
            num_D_constraints = self.r_number_of_records * (amo_D_extra_constraints + 1)

            # num_extra_vars = num_R_extra_vars + num_D_extra_vars + num_qeq_extra_vars + num_rid_extra_vars 
            num_extra_vars = num_D_extra_vars + num_rid_extra_vars
            num_total_vars = num_vars + num_extra_vars 

            # num_total_constraints = num_R_constraints + num_D_constraints + num_qeq_constraints + num_rid_constraints
            num_total_constraints = num_D_constraints + num_rid_constraints

            # num_total_vars = num_vars + num_R_extra_vars + num_D_extra_vars + num_qeq_extra_vars
            # num_total_constraints = num_R_constraints + num_D_constraints + num_qeq_constraints

            # print(num_R_vars, num_D_vars, num_R_extra_vars, num_D_extra_vars, num_qeq_extra_vars, num_rid_extra_vars)
            # print(num_R_constraints, num_D_constraints, num_qeq_constraints, num_rid_constraints) 

            # Construct query matrix.
            #print("Constructing Q...")
            f = open(self.file_name, "w+")

            f.write('p cnf' + " " + str(num_total_vars) + " " + str(num_total_constraints) + " \n")
            # f.write('p cnf' + " " + str(num_total_vars_prev) + " " + str(num_Q_constraints_prev + num_D_constraints + num_R_constraints) + " \n")

            # print("Constructing Q...")
            # start = time.perf_counter_ns()
            
            # for i in range(self.t_number_of_queries): 
            #     R_vars = []
            #     for j in range(len(self.set_of_ranges)): 
            #         R_vars.append(R_matrix[i][j])
            #     write_pbeq_clause(R_vars,R_extra_vars[i],f)

            # end = time.perf_counter_ns()
            
            # print(f"Elapsed: {(end - start) / (10 ** 9)} s")
            print("Constructing D...")
            start = time.perf_counter_ns()
            D_matrix = [[0 for _ in range(self.r_number_of_records)] for _ in range(self.n_domain_size)]
            for i in range(self.n_domain_size): 
                for j in range(self.r_number_of_records): 
                    D_matrix[i][j] = i*self.r_number_of_records + j + 1

            for i in range(self.r_number_of_records):
                column = []
                for j in range(self.n_domain_size):
                    column.append(D_matrix[j][i])
                
                # exactly one of these are true 
                write_pbeq_clause(column, D_extra_vars[i], f) 

            # R_matrix_extra_vars = {} 
            # num_prev_vars = num_vars + num_Q_extra_vars + num_D_extra_vars
            # for i in range(R_num_ones): 
            #     R_matrix_extra_vars[i] =[(num_prev_vars + self.n_domain_size* i + j) for j in range(1,self.n_domain_size + 1)]

            end = time.perf_counter_ns()
            print(f"Elapsed: {(end - start) / (10 ** 9)} s")

            for i in range(len(rid_leakage)): 
                B = rid_leakage[i]
                for j in range(len(B)): 
                    matches = T[query_sequence[i]][j]
                    # print(matches)
                    for k in range(len(B[0])): 
                        points = [D_matrix[x][k] for x in matches]
                        # print("points")
                        # print([str(x) for x in points])
                        if B[j][k]: 
                            f.write(" ".join([str(x) for x in points]) + " 0\n")
                        else: 
                            for point in points: 
                                f.write(str(-1 * point) + " 0\n")
    
            f.close()
            end = time.perf_counter_ns()
            
            # print(f"Elapsed: {(end - start) / (10 ** 9)} s")
            # print("Solving ...")
            start = time.perf_counter_ns() 
            cmd = './build/cadical ' +  self.file_name + ' > testing.txt'
            os.system(cmd) 

            recovered_D_matrix = parse_output_file('testing.txt', D_matrix, self.n_domain_size, self.r_number_of_records, num_vars)

            if recovered_D_matrix != -1: 
            # end = time.perf_counter_ns()
            
            # print(f"Elapsed: {(end - start) / (10 ** 9)} s")
                print(recovered_D_matrix)
                return recovered_D_matrix

def compute_mae(data,recovered,domain_size): 
    errors = np.absolute([float(recovered[x] - data[x])/domain_size for x in range(len(data))])
    return float(sum(errors)) / len(errors)

def compute_recovery_rates(matrix, recovered_matrix): 
    r_number_of_records = len(matrix)
    n_domain_size = len(matrix[0]) 

    matrix_idx = [np.nonzero(x)[0][0] for x in matrix]
    recovered_matrix_idx = [np.nonzero(x)[0][0] for x in recovered_matrix]
    # recovered_matrix_idx_reversed = [(n_domain_size - 1) - x for x in recovered_matrix_idx]

    print(matrix_idx)
    print(recovered_matrix_idx)

    num_wrong_left = np.count_nonzero(np.sum(np.abs(recovered_matrix - matrix),axis=1))
    # num_wrong_right = np.count_nonzero(np.sum(np.abs(np.flip(recovered_matrix,axis=1) - matrix),axis=1))

    recovery_rate = (r_number_of_records - num_wrong_left)/float(r_number_of_records)

    # Compute approximation metric here
    approx_metric_left = sum([abs(i - j) for i,j in zip(matrix_idx, recovered_matrix_idx)])
    #  approx_metric_right = sum([abs(i - j) for i,j in zip(matrix_idx, recovered_matrix_idx_reversed)])

    approx_recovery_rate = 1 - approx_metric_left/(math.floor(n_domain_size * r_number_of_records))

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
    set_of_edges, range_to_min_cover, _, _ = compute_hypergraph_info(n_domain_size, set_of_ranges)
    # print(set_of_ranges) 
    # Generating a random Q matrix 
    if query_dist == "zipf": 
        distribution = Zipfian(number_of_ranges, 2) # add alpha to parameters
    elif query_dist == "uniform": 
        distribution = Uniform(number_of_ranges)
    else:
        print(query_dist)

    queries = distribution.sample(t_number_of_queries) 
    print("QUERIES")
    # queries = [6,22,21,8,24,20,26,24]
    # queries = [14, 29, 28, 16, 28, 31, 30, 4]
    queries = [38,59,85,96,13,52,36,39,55,102,35,56,16,50,114,130]
    # queries = [11,4,88,82,75,34,25,37,29,84,1,11,95,82,47,36]
    # queries = [58,132,61,1,123,15,103,57,45,121,50,5,4,122,54,80]
    # queries = [2,30,11,23,25,21,23,6,11,14,25,10,23,19,5,13,12,15,29,13,7,31,18,25,29,15,26,30,30,4,14,22]
    print(queries)
    # Generating a random D matrix (using uniform distribution)
    D_matrix = [[0 for _ in range(n_domain_size)] for _ in range(r_number_of_records)] 

    if data_dist == "fixed_density": 
        uniform_samples = FixedDensity(n_domain_size,density_pct).sample(r_number_of_records)
    elif data_dist == "uniform": 
        uniform_samples = Uniform(n_domain_size).sample(r_number_of_records)
    # uniform_samples = [1,3,2,0]
    # uniform_samples = [11,13,8,6,11,10,10,2,9,14,8,3,11,12,12,5]
    # uniform_samples = [28,25,26,8,15,20,8,2,2,9,11,28,2,3,2,29,10,26,29,0,22,30,17,29,13,10,23,15,22,30,9,28]
    for i in range(r_number_of_records): 
        D_matrix[i][uniform_samples[i]] = 1
    # print(uniform_samples)
    # print(queries, uniform_samples)
    # Compute the OST Leakage from Q_matrix and D_matrix 
    qeq_leakage = gen_qeq_leakage(queries, range_to_min_cover) 
    # print(qeq_leakage)
    rid_leakage = gen_rid_leakage(queries, uniform_samples, set_of_edges, range_to_min_cover) 

    ostsolver = OSTLeakageSolver(
        t_number_of_queries=t_number_of_queries,
        n_domain_size=n_domain_size,
        r_number_of_records=r_number_of_records,
        file_name = "test.cnf"
    )

    # recovered_D_matrix = ostsolver.solve(qeq_leakage = qeq_leakage, rid_leakage = rid_leakage)

    tmp = ostsolver.compute_all_possible_range_matrices(qeq_leakage) 
    recovered_D_matrix = ostsolver.solve(tmp, rid_leakage)

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
    f = open("fixed_density_t100_n10_r10.txt", "w")
    t_list = [32]
    n_list = [32]
    r_list = [32]
    data_dist = "uniform" #uniform or fixed_density for now
    query_dist = "uniform" #uniform or zipf
    num_iters = 1
    num_threads = 1

    for t_number_of_queries in t_list: 
        for n_domain_size in n_list: 
            for r_number_of_records in r_list: 

                with Pool(num_threads) as p: 
                    if data_dist == "fixed_density": 
                        densities = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
                    else: 
                        densities = [1]
                    for density_pct in densities:
                        start = time.time()

                        results = p.starmap(run_one_instance, [(t_number_of_queries,n_domain_size,r_number_of_records, data_dist, query_dist, density_pct,) for _ in range(num_iters)])
                        averages = np.array(results).sum(axis=0) / num_iters
                
                        end = time.time()
                
                        print(averages)
                        f.write(str(t_number_of_queries) + ", " + str(n_domain_size) + ", " + str(r_number_of_records) + ", " + ", ".join(str(round(x, 2)) for x in averages))
                        f.write("\n")
    f.close()

if __name__ == "__main__":
    start = time.perf_counter_ns()
    # compute_hypergraph_info(n_domain_size,set_of_ranges)
    print("HELLO")
    main()
    # gen_set_of_edges(8)
    # print(compute_D_extra_info(200))
    # run_one_instance(100,100,100,"fixed_density",'uniform',1)
    end = time.perf_counter_ns()
    print(f"Elapsed: {(end - start) / (10 ** 9)} s")