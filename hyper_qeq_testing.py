from typing import List
from z3 import *
from copy import deepcopy

from dataclasses import dataclass
from multiprocessing import Pool
from collections import defaultdict

import random
import time
import math

import numpy as np
import pickle
from scipy.stats import zipfian, randint

import operator
import functools

from typing import Any
from queue import PriorityQueue
import heapq
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

    index = []
    for i in range(len(output)): 
        if output[i]: 
            index.append(i) 

    return output, index 

def gen_range_to_min_cover(set_of_ranges, n_domain_size): 
    output = {}
    indices = {} 
    for i in range(len(set_of_ranges)): 
        r = set_of_ranges[i]
        output[i], indices[i] = gen_range_min_cover(r, n_domain_size)

    return output, indices 

def compute_hypergraph_info(n_domain_size, set_of_ranges):
    s = 2*n_domain_size - 1 
    set_of_edges = gen_set_of_edges(n_domain_size) 
    range_to_min_cover, range_to_min_cover_indices = gen_range_to_min_cover(set_of_ranges, n_domain_size) 

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

    return set_of_edges, range_to_min_cover, range_to_min_cover_indices, E_matrix, list_of_H_matrix

def gen_qeq_leakage(queries, range_to_min_cover_indices): 
    output = {}
    t = len(queries)
    for i in range(t): 
        for j in range(i + 1,t): 
            r1 = range_to_min_cover_indices[queries[i]]
            r2 = range_to_min_cover_indices[queries[j]]
            overlap = gen_ranges_overlap(r1,r2) 
            output[(i,j)] = (len(r1), len(r2), overlap)

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

def gen_ranges_overlap(range1,range2): 
    i,j,m,n = 0,0, len(range1), len(range2) 
    output = [] 
    count = 0
    while i < m and j < n: 
        if range1[i] < range2[j]: 
            count += 1 
            i += 1
        elif range1[i] > range2[j]: 
            count += 1 
            j += 1 
        else: 
            output.append(count) 
            count += 1 
            i += 1 
            j += 1
    return output 

def gen_S_sets(H_matrices, range_to_min_cover, range_to_min_cover_indices): 
    output = {}
    num_ranges = len(H_matrices[0])
    for i in range(num_ranges): 
        for j in range(i, num_ranges): 
            alpha = sum(range_to_min_cover[i])
            beta = sum(range_to_min_cover[j])
            lst1 = range_to_min_cover_indices[i]
            lst2 = range_to_min_cover_indices[j]

            overlap = gen_ranges_overlap(lst1,lst2)

            if (alpha,beta) in output: 
                output[(alpha,beta)][tuple(overlap)].add((i,j))
            else: 
                output[(alpha,beta)] = defaultdict(set)
                output[(alpha,beta)][tuple(overlap)].add((i,j))
            
            if i != j: 
                if (beta,alpha) in output: 
                    output[(beta,alpha)][tuple(overlap)].add((j,i))
                else: 
                    output[(beta,alpha)] = defaultdict(set)
                    output[(beta,alpha)][tuple(overlap)].add((j,i))
    
    return output 

def gen_T_sets(H_matrices, E_matrix): 
    output = {} 
    n = len(H_matrices) 
    num_ranges = len(H_matrices[0])
    N = len(E_matrix) 
    for j in range(n): 
        range_to_edge_mapping = defaultdict(list)
        for alpha in range(num_ranges): 
            for x in range(N): 
                row1 = H_matrices[j][alpha]
                row2 = E_matrix[x]
                if sum([a * b for a,b in zip(row1,row2)]) == 1: 
                    range_to_edge_mapping[alpha].append(x)
        output[j] = range_to_edge_mapping
    return output    

def compute_qeq_extra_vars(S, R_vars, possible_query_output, qeq_leakage, t, curr_num): 
    extra_vars = 0
    extra_constraints = 0
    output = defaultdict(list)
    not_checked_indices = []
    for i in range(t): 
        if len(possible_query_output[i]) == 1 and -1 in possible_query_output[i]: # need to manually enumerate the constraints here 
            not_checked_indices.append(i) 
    
    visited = set() 
    for i in range(t): 
        for j in range(i + 1, t): 
            if i in not_checked_indices or j in not_checked_indices: 
                if (i,j) not in visited: 
                    visited.add((i,j))
                    a,b,A = qeq_leakage[(i,j)]  
                    matches = S[(a,b)][tuple(A)] 
                    cleaned_matches = [] 
                    for match in matches: 
                        x,y = match 
                        if R_vars[i][x] > 0 and R_vars[j][y] > 0: 
                            cleaned_matches.append(match)

                    output[(i,j)] = [curr_num + 1 + x for x in range(len(cleaned_matches))]
                    extra_vars += len(cleaned_matches)
                    extra_constraints += 2 * len(cleaned_matches) + 1
                    curr_num += len(cleaned_matches) 
    
    return output, extra_vars, extra_constraints 

def compute_rid_extra_vars(T, possible_query_output, D_vars, R_vars, rid_leakage, t, curr_num): 
    extra_vars = 0
    extra_constraints = 0
    output = {}
    T_prime = {} 
    num_ranges = len(R_vars[0])
    for i in range(t): 
        B = rid_leakage[i] 
        num_possible = np.count_nonzero(R_vars[i])
        for j in range(len(B)): 
            matches = [] 
            for r in range(num_ranges): 
                if R_vars[i][r] > 0: 
                    if num_possible == 1: 
                        for match in T[j][r]: 
                            matches.append(match)
                    else: 
                        for match in T[j][r]: 
                            matches.append((r,match))
            T_prime[(i,j)] = matches 
            for k in range(len(B[0])): 
                if num_possible == 1: 
                    if B[j][k]: 
                        output[(i,j,k)] = [] 
                        extra_constraints += 1 
                    else: 
                        output[(i,j,k)] = []
                        for x in T_prime[(i,j)]: 
                            if D_vars[x][k] > 0:
                                extra_constraints += 1
                else: 
                    num_matches = 0
                    for _,x in T_prime[(i,j)]: 
                        if D_vars[x][k] > 0: 
                            num_matches += 1
                    if B[j][k]: 
                        output[(i,j,k)] = [curr_num + 1 + x for x in range(num_matches)]
                        extra_vars += num_matches
                        extra_constraints += 2 * num_matches + 1
                        curr_num += num_matches
                    else: 
                        output[(i,j,k)] = []
                        extra_constraints += num_matches

    return output, T_prime, extra_vars, extra_constraints 

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
            if D_matrix_mapping[j][i] != 0: 
                output[i][j] = tmp[D_matrix_mapping[j][i]]
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

def compute_R_info(possible_query_output, set_of_ranges, t): 
    num_ranges = len(set_of_ranges) 
    curr = 0 
    R_vars =  [[0 for _ in range(num_ranges)] for _ in range(t)]
    for i in range(t):
        possible = possible_query_output[i]
        if -1 in possible: 
            for j in range(num_ranges): 
                R_vars[i][j] = curr + 1 
                curr += 1
        else: 
            for j in possible: 
                R_vars[i][j] = curr + 1 
                curr += 1
    R_extra_vars = {} 
    num_R_constraints = 0 

    for i in range(t): 
        extra_vars, extra_constraints = compute_amo_extra_info(np.count_nonzero(R_vars[i]))
        R_extra_vars[i] = [curr + j + 1 for j in range(extra_vars)]
        curr += extra_vars
        num_R_constraints += (extra_constraints + 1)
    
    test_vars, test_constraints = compute_amo_extra_info(num_ranges)
    return R_vars, R_extra_vars, curr, num_R_constraints, (t * test_vars) + (t * num_ranges), (t * (test_constraints + 1))

def compute_D_info(n, r, candidate_D, curr): 
    num_D_vars = 0
    # amo_D_vars, amo_D_constraints = compute_amo_extra_info(n)
    D_vars = [[0 for _ in range(r)] for _ in range(n)]
    for i in range(n): 
        for j in range(r): 
            if candidate_D[i][j]: 
                D_vars[i][j] = curr + 1
                curr += 1
                num_D_vars += 1
    D_extra_vars = {} 
    num_D_constraints = 0
    for i in range(r):
        test = [D_vars[x][i] for x in range(n)]
        extra_vars, extra_constraints = compute_amo_extra_info(np.count_nonzero(test))
        D_extra_vars[i] = [(curr + j) for j in range(1, extra_vars + 1)]
        curr += extra_vars 
        num_D_vars += extra_vars
        num_D_constraints += (extra_constraints + 1)
    return D_vars, D_extra_vars, num_D_vars, num_D_constraints 

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

        self.set_of_edges,self.range_to_min_cover,self.range_to_min_cover_indices,self.E_matrix, self.list_of_H_matrix = compute_hypergraph_info(self.n_domain_size,self.set_of_ranges)
        print("gen S sets")
        start = time.perf_counter_ns()
        if self.n_domain_size > 32: 
            S_file_name = "pkl/S_" + str(self.n_domain_size) + ".pkl"
            S_file = open(S_file_name, 'rb')
            self.S = pickle.load(S_file) 
            S_file.close() 
            end = time.perf_counter_ns()
        else:
            self.S = gen_S_sets(self.list_of_H_matrix,self.range_to_min_cover,self.range_to_min_cover_indices)
        end = time.perf_counter_ns()
        print(f"Gen S Sets: {(end - start) / (10 ** 9)} s")
        print("gen T sets")
        start = time.perf_counter_ns()
        if self.n_domain_size > 32: 
            T_file_name = "pkl/T_" + str(self.n_domain_size) + ".pkl"
            T_file = open(T_file_name, 'rb')
            self.T = pickle.load(T_file) 
            T_file.close
        else: 
            self.T = gen_T_sets(self.list_of_H_matrix, self.E_matrix)
        end = time.perf_counter_ns()
        print(f"Gen T Sets: {(end - start) / (10 ** 9)} s")
    
    def check_is_valid_rid_query(self, range_query_num, B, prev_data): 
        data = [row[:] for row in prev_data]
        # print(prev_data)
        for i in range(len(B)): 
            candidates = tuple(self.T[i][range_query_num])
            # print(range_query_num, i)
            # print(candidates)
            for j in range(len(B[0])): 
                if B[i][j] == 1: 
                    for k in range(len(data)): 
                        if data[k][j] == -1: 
                            if k in candidates: 
                                data[k][j] = 1
                            else: 
                                data[k][j] = 0
                        elif data[k][j] == 1: 
                            if k not in candidates: 
                                data[k][j] = 0
                else: 
                    for k in range(len(data)): 
                        if data[k][j] == -1: 
                            if k in candidates: 
                                data[k][j] = 0
                            else: data[k][j] = 1
                        elif data[k][j] == 1:
                            if k in candidates: 
                                data[k][j] = 0
                            else: 
                                check = True or check
        test = [[data[x][i] for x in range(len(data))] for i in range(len(data[0]))]
        if all([sum(test[i]) >= 1] for i in range(len(test))):  
            return True, data 
        else: 
            return False, [] 
    
    def check_is_valid_rid_query_sequence(self, query_sequence, rid_leakage): 
        D = [[-1 for _ in range(self.r_number_of_records)] for _ in range(self.n_domain_size)]
        for i in range(len(query_sequence)): 
            query = query_sequence[i]
            B = rid_leakage[i] 
            if query != -1: 
                is_valid, D = self.check_is_valid_rid_query(query, B, D)   
            if not is_valid: 
                return False, []
        
        return True, D 
    
    def compute_all_possible_range_matrices(self, qeq_leakage, rid_leakage): 

        candidates = []
        D = [[-1 for _ in range(self.r_number_of_records)] for _ in range(self.n_domain_size)]
   
        # print(qeq_leakage)
        start = time.perf_counter_ns()
        candidate_matches = {} 
        for i in range(self.t_number_of_queries):
            for j in range(i + 1,self.t_number_of_queries):
                a,b,A = qeq_leakage[(i,j)]
                
                matches = self.S[(a,b)][tuple(A)]
                candidate_matches[(i,j)] = matches
                heapq.heappush(candidates, (len(matches),i,j))

        # candidate_ranges = [([-1 for _ in range(self.t_number_of_queries)], D)]
        candidate_ranges = [[-1 for _ in range(self.t_number_of_queries)]]

        visited = set()
        t_end = time.time() + 1
        while candidates and time.time() < t_end: 
            _,i,j = heapq.heappop(candidates)
            if (i in visited) and (j in visited): # we've already checked 
                continue
            old_candidate_ranges = [deepcopy(tup) for tup in candidate_ranges]
            new_candidate_ranges = []

            matches = candidate_matches[(i,j)]
            print(len(matches))
            preprocessed_matches_i = defaultdict(set)
            preprocessed_matches_j = defaultdict(set) 
            for match in matches: 
                x,y = match 
                preprocessed_matches_i[x].add(y)
                preprocessed_matches_j[y].add(x)

            for candidate_range in old_candidate_ranges: 
                if candidate_range[i] == -1 and candidate_range[j] == -1: # both are currently unset 
                    for match in matches: 
                        tmp = candidate_range.copy() 
                        # old_data = D.copy()
                        # print(old_data)
                        tmp[i] = match[0] # set query in candidate query sequence 
                        tmp[j] = match[1] 
                        to_append = True 
                        # check whether can be valid rid sequence 
                        # to_append, D_prime = self.check_is_valid_rid_query(match[0], rid_leakage[i], D)
                        # if to_append: 
                        #     to_append, new_D = self.check_is_valid_rid_query(match[1],rid_leakage[j],D_prime)
                        if to_append: 
                            pairs = [(x,y) for x in visited for y in [i,j]] # check that this can be a valid hyperqeq sequence 
                            for x,y in pairs: 
                                pair = (x,y) if x < y else (y,x)  
                                check = candidate_matches[pair]
                                if (tmp[pair[0]], tmp[pair[1]]) not in check: 
                                    to_append = False 
                                    break 
                        if to_append: 
                            new_candidate_ranges.append(tmp) 
                else: 
                    if candidate_range[i] > 0: 
                        matches = preprocessed_matches_i[candidate_range[i]]
                        for match in matches: 
                            tmp = candidate_range.copy()
                            tmp[j] = match
                            to_append = True 
                            # to_append, D_prime = self.check_is_valid_rid_query(match, rid_leakage[j], D)
                            if to_append: 
                                pairs = [(x,j) for x in visited]
                                for x,y in pairs: 
                                    pair = (x,y) if x < y else (y,x)  
                                    check = candidate_matches[pair]
                                    if (tmp[pair[0]], tmp[pair[1]]) not in check: 
                                        to_append = False 
                                        break 
                            if to_append: 
                                new_candidate_ranges.append(tmp) 
                    elif candidate_range[j] > 0: 
                        matches = preprocessed_matches_j[candidate_range[j]]
                        for match in matches: 
                            tmp = candidate_range.copy() 
                            tmp[i] = match
                            to_append = True 
                            # to_append, D_prime = self.check_is_valid_rid_query(match, rid_leakage[i], D)
                            if to_append: 
                                pairs = [(x,i) for x in visited]
                                for x,y in pairs: 
                                    pair = (x,y) if x < y else (y,x)  
                                    check = candidate_matches[pair]
                                    if (tmp[pair[0]], tmp[pair[1]]) not in check: 
                                        to_append = False 
                                        break 
                            if to_append: 
                                new_candidate_ranges.append(tmp) 
            candidate_ranges = new_candidate_ranges.copy()
            visited.add(i)
            visited.add(j)
            print(len(visited), len(candidate_ranges))

        output = {x:set() for x in range(self.t_number_of_queries)}
        D = [[-1 for _ in range(self.r_number_of_records)] for _ in range(self.n_domain_size)]
        # curr = 0
        for candidate_range in candidate_ranges: 

            # check if is a valid sequence when comparing to rid leakage 
            # is_valid, D_prime = self.check_is_valid_rid_query_sequence(candidate_range, rid_leakage)
            # if is_valid: 
            #     print(curr) 
            #     curr += 1
            for i in range(len(candidate_range)): 
                output[i].add(candidate_range[i])
            #     for i in range(len(D)): 
            #         for j in range(len(D[0])): 
            #             D[i][j] = D[i][j] or D_prime[i][j]
            
            # for i in range(self.n_domain_size): 
            #     for j in range(self.r_number_of_records): 
            #         D[i][j] = D[i][j] or candidate_D[i][j]
        
        test = [] 
        for key, val in output.items(): 
            if len(val) == 1 and -1 not in val: 
                test.append(key) 
        print('this is test')
        print(test)
        for key in test: 
            _, D = self.check_is_valid_rid_query(list(output[key])[0], rid_leakage[key], D) 

        return output, D


    def solve(self,possible_query_output,candidate_D,qeq_leakage,rid_leakage):

        # Preparing all the boolean variables for cadical 

        R_vars, R_extra_vars, num_R_vars, num_R_constraints, test_vars, test_constraints = compute_R_info(possible_query_output, self.set_of_ranges, self.t_number_of_queries)
        D_vars, D_extra_vars, num_D_vars, num_D_constraints = compute_D_info(self.n_domain_size, self.r_number_of_records, candidate_D, num_R_vars) 

        qeq_extra_vars, num_qeq_extra_vars, num_qeq_constraints = compute_qeq_extra_vars(self.S, R_vars, possible_query_output, qeq_leakage, self.t_number_of_queries, num_R_vars + num_D_vars)
        rid_extra_vars, T_prime, num_rid_extra_vars, num_rid_constraints = compute_rid_extra_vars(self.T, possible_query_output, D_vars, R_vars, rid_leakage, self.t_number_of_queries, num_R_vars + num_D_vars + num_qeq_extra_vars)

        num_vars = num_R_vars + num_D_vars 
        num_extra_vars = num_qeq_extra_vars + num_rid_extra_vars 
        num_total_vars = num_vars + num_extra_vars 
        num_constraints = num_R_constraints + num_D_constraints + num_qeq_constraints + num_rid_constraints 

        print(num_R_vars, num_D_vars, num_R_constraints, num_D_constraints)
        print(num_R_constraints, num_D_constraints, num_qeq_constraints, num_rid_constraints) 

        f = open(self.file_name, "w+")
        f.write('p cnf' + " " + str(num_total_vars) + " " + str(num_constraints) + " \n")

        # pbeq constraints for R matrix 

        for i in range(self.t_number_of_queries): 
            booleans  = [] 
            for j in range(len(self.set_of_ranges)): 
                if R_vars[i][j] > 0: 
                    booleans.append(R_vars[i][j])
            write_pbeq_clause(booleans,R_extra_vars[i], f)

        #pbeq constraints for D matrix 

        for i in range(self.r_number_of_records):
            column = []
            for j in range(self.n_domain_size):
                if D_vars[j][i] > 0: 
                    column.append(D_vars[j][i])
            
            # exactly one of these are true 
            write_pbeq_clause(column, D_extra_vars[i], f) 

        # # writing QEQ leakage 

        not_checked_indices = []
        for i in range(self.t_number_of_queries): 
            if len(possible_query_output[i]) == 1 and -1 in possible_query_output[i]: 
                not_checked_indices.append(i) 
        visited = set() 
        for i in range(self.t_number_of_queries): 
            for j in range(i + 1, self.t_number_of_queries): 
                if i in not_checked_indices or j in not_checked_indices: 
                    if (i,j) not in visited: 
                        visited.add((i,j))
                        a,b,A = qeq_leakage[(i,j)]
                        matches = self.S[(a,b)][tuple(A)]
                        row = []
                        col = [] 
                        for match in matches: 
                            x,y = match 
                            if R_vars[i][x] > 0 and R_vars[j][y] > 0: 
                                row.append(R_vars[i][match[0]])
                                col.append(R_vars[j][match[1]])
                        write_leakage_clauses_to_file(row,col,qeq_extra_vars[(i,j)],1,f)

        # # Writing RID leakage constraints 

        for i in range(self.t_number_of_queries): 
            B = rid_leakage[i] 
            num_possible = np.count_nonzero(R_vars[i])
            for j in range(len(B)): 
                tmp = T_prime[(i,j)]
                for k in range(len(B[0])): 
                    if num_possible == 1: 
                        points = [] 
                        for x in tmp: 
                            if D_vars[x][k] > 0: 
                                points.append(D_vars[x][k])
                        if B[j][k]: 
                            f.write(" ".join([str(x) for x in points]) + " 0\n")
                        else: 
                            for x in points: 
                                f.write(str(-1 * x) + " 0\n")
                    else: 
                        matches = []
                        for r,x in tmp: 
                            if D_vars[x][k] > 0: 
                                matches.append((r,x))
                        row = []
                        col = []
                        for match in matches: 
                            row.append(D_vars[match[1]][k])
                            col.append(R_vars[i][match[0]])
                        
                        write_leakage_clauses_to_file(row,col,rid_extra_vars[(i,j,k)], B[j][k], f)
        
        f.close()
        print("Solving ...")
        start = time.perf_counter_ns() 
        cmd = './build/cadical ' +  self.file_name + ' > testing.txt'
        os.system(cmd) 

        recovered_D_matrix = parse_output_file('testing.txt', D_vars, self.n_domain_size, self.r_number_of_records, num_R_vars + num_D_vars)
        # print(recovered_D_matrix)

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
    set_of_edges, range_to_min_cover, range_to_min_cover_indices, _, _ = compute_hypergraph_info(n_domain_size, set_of_ranges)
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
    # queries = [31,9,10,4,26,10,22,14]
    # queries = [6,22,21,8,24,20,26,24]
    # queries = [14, 29, 28, 16, 28, 31, 30, 4]
    # queries = [38,59,85,96,13,52,36,39,55,102,35,56,16,50,114,130]
    # queries = [11,4,88,82,75,34,25,37,29,84,1,11,95,82,47,36]
    # queries = [58,132,61,1,123,15,103,57,45,121,50,5,4,122,54,80] # 1015680 options
    # queries = [2,30,11,23,25,21,23,6,11,14,25,10,23,19,5,13,12,15,29,13,7,31,18,25,29,15,26,30,30,4,14,22]
    # queries = [256,91,341,155,32,10,167,391,337,230,186,191,156,344,202,18,495,422,466,228,145,405,92,380,15,406,1,137,217,186,421,79]
    print(queries)
    # Generating a random D matrix (using uniform distribution)
    D_matrix = [[0 for _ in range(n_domain_size)] for _ in range(r_number_of_records)] 

    if data_dist == "fixed_density": 
        uniform_samples = FixedDensity(n_domain_size,density_pct).sample(r_number_of_records)
    elif data_dist == "uniform": 
        uniform_samples = Uniform(n_domain_size).sample(r_number_of_records)
    elif data_dist == "zipf": 
        uniform_samples = Zipfian(n_domain_size, s=2).sample(r_number_of_records)
    # uniform_samples = [1,3,2,0]
    # uniform_samples = [3,5,0,6,3,2,2,1]
    # uniform_samples = [11,13,8,6,11,10,10,2,9,14,8,3,11,12,12,5]
    # uniform_samples = [28,25,26,8,15,20,8,2,2,9,11,28,2,3,2,29,10,26,29,0,22,30,17,29,13,10,23,15,22,30,9,28]
    for i in range(r_number_of_records): 
        D_matrix[i][uniform_samples[i]] = 1
    print(uniform_samples)
    # print(queries, uniform_samples)
    # Compute the OST Leakage from Q_matrix and D_matrix 
    qeq_leakage = gen_qeq_leakage(queries, range_to_min_cover_indices)
    # print(qeq_leakage)
    # print(qeq_leakage)
    rid_leakage = gen_rid_leakage(queries, uniform_samples, set_of_edges, range_to_min_cover) 

    ostsolver = OSTLeakageSolver(
        t_number_of_queries=t_number_of_queries,
        n_domain_size=n_domain_size,
        r_number_of_records=r_number_of_records,
        file_name = "test.cnf"
    )


    possible_queries, candidate_D = ostsolver.compute_all_possible_range_matrices(qeq_leakage, rid_leakage) 
    recovered_D_matrix = ostsolver.solve(possible_queries, candidate_D, qeq_leakage, rid_leakage)

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
    f = open("results.txt", "w")
    t_list = [32]
    n_list = [8]
    r_list = [8]
    data_dist = "uniform" #uniform or fixed_density for now
    query_dist = "zipf" #uniform or zipf
    num_iters = 10
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
    print("HELLO")
    main()
    end = time.perf_counter_ns()
    print(f"Elapsed: {(end - start) / (10 ** 9)} s")
    # n_domain_size = 128
    # set_of_ranges = []
    # for i in range(n_domain_size):
    #     for j in range(i, n_domain_size):
    #             set_of_ranges.append((i,j))

    # set_of_edges,range_to_min_cover,range_to_min_cover_indices,E_matrix,list_of_H_matrix = compute_hypergraph_info(n_domain_size,set_of_ranges)
    # # S = gen_S_sets(list_of_H_matrix,range_to_min_cover,range_to_min_cover_indices)

    # T = gen_T_sets(list_of_H_matrix, E_matrix)

    # f = open("pkl/T_128.pkl", 'wb')
    # pickle.dump(T,f)
    # f.close()
