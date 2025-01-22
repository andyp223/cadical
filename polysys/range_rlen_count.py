from typing import List
from z3 import *

from dataclasses import dataclass
import pickle

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

def bit_decomposition(n):
    """Returns the bit decomposition of an integer n as a list."""
    if n < 0:
        raise ValueError("Input must be a non-negative integer.")
    
    bits = []
    while n > 0:
        if n & 1:  # Check if the least significant bit is 1
            bits.append(1)  # Add 2^position to the result
        else: 
            bits.append(0)
        n >>= 1  # Right shift the number by 1 bit
    return bits


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

def compute_D_info(t, r, n, q): 
    num_vars_log_r = len(bit_decomposition(r))
    len_log_n = len(bit_decomposition(n))

    D_vars = [] 
    for i in range(2 ** len_log_n): 
        tmp = [q + 1 + j for j in range(num_vars_log_r)]
        D_vars.append(tmp)
        q += num_vars_log_r 

    max_val = (2 ** num_vars_log_r) - 1
    num_constraints = max_val - r

    return D_vars, (2 ** len_log_n) * num_vars_log_r , n * num_constraints + (2 ** len_log_n - n) * num_vars_log_r

def compute_sum_extra_vars(log_r, n, curr_num): 
    sum_vars = {} 
    carry_vars = {} 
    carry_extra_vars = {} 

    curr_round = 0 
    round_size = n // 2 

    while round_size > 0: 
        tmp_sum_vars = []
        tmp_carry_vars = []
        tmp_carry_extra_vars = [] 

        for i in range(round_size): 
            tmp_sum_vars.append([curr_num + 1 + i for i in range(log_r)])
            curr_num += log_r 
            tmp_carry_vars.append([curr_num + 1 + i for i in range(log_r)])
            curr_num += log_r 
            tmp_carry_extra_vars.append([curr_num + 1 + i for i in range(log_r - 1)])
            curr_num += log_r - 1 
        
        sum_vars[curr_round] = tmp_sum_vars 
        carry_vars[curr_round] = tmp_carry_vars
        carry_extra_vars[curr_round] = tmp_carry_extra_vars
        curr_round += 1
        round_size //=2 
        log_r += 1 

    # TODO: Compute number of constraints 
    return sum_vars, carry_vars, carry_extra_vars, curr_num 

def compute_leakage_constraints(t,r,n,leakage,curr_num): 
    num_vars_log_r = len(bit_decomposition(r))
    max_val = (2 ** num_vars_log_r) - 1

    num_extra_vars = 0
    num_extra_constraints = 0 

    count_vars = {} 

    for i in range(t): 
        for j in range(n): 
            for k in range(num_vars_log_r): 
                count_vars[(i,j,k)] = curr_num + 1
                curr_num += 1
                num_extra_vars += 1
                num_extra_constraints += 3
    
    sum_vars = {}
    carry_vars = {} 
    carry_extra_vars = {} 
    for i in range(t + 1): 
        # need to sum n log_r bit numbers together 
        tmp_sum_vars, tmp_carry_vars, tmp_carry_extra_vars, curr_num = compute_sum_extra_vars(num_vars_log_r, 2 ** len(bit_decomposition(n)), curr_num)

        sum_vars[i] = tmp_sum_vars 
        carry_vars[i] = tmp_carry_vars
        carry_extra_vars[i] = tmp_carry_extra_vars

    num_extra_constraints = 0
    return count_vars,sum_vars,carry_vars,carry_extra_vars,num_extra_vars,num_extra_constraints, curr_num

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
    a = len(D_matrix_mapping)
    b = len(D_matrix_mapping[0])
    output = [[0 for _ in range(len(D_matrix_mapping[0]))] for _ in range(len(D_matrix_mapping))]
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

    for i in range(a): 
        for j in range(b): 
            if D_matrix_mapping[i][j] != 0: 
                output[i][j] = tmp[D_matrix_mapping[i][j]]
    
    recovered_counts = [] 
    for i in range(n): 
        num = sum(bit * (2 ** idx) for idx, bit in enumerate(output[i]))
        recovered_counts.append(num) 
    return recovered_counts

# copied from LEAKER 
def compute_recovery_rates(a, b): 
    vals = sorted(a, reverse=True)
    recovered_vals = sorted(b, reverse=True) 

    # while len(recovered) < len(s_vals):
    #     recovered.append(0)

    # while len(s_vals) < len(recovered):
    #     s_vals.append(0)

    assert len(vals) == len(recovered_vals)

    err = 0
    for i in range(len(vals)):
        if vals[i] == 0 and recovered_vals[i] == 0:
            err += 0
        else:
            err += (abs(vals[i] - recovered_vals[i]) / max(vals[i], recovered_vals[i]))
            # err += 1 - min(vals.count(s_vals[i]), recovered[i]) / max(vals.count(s_vals[i]),
            #                                                             recovered[i])
    err /= len(vals) 
    return err

class RangeRlenCountSolver:
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
        D_vars, num_D_vars, num_D_constraints = compute_D_info(self.t_number_of_queries, self.r_number_of_records, self.n_domain_size, num_Q_vars)

        rid_vars,sum_vars,carry_vars,carry_extra_vars,num_leakage_vars,num_leakage_constraints, total_vars = compute_leakage_constraints(self.t_number_of_queries, self.r_number_of_records, self.n_domain_size, leakage, num_Q_vars+num_D_vars)

        f = open(self.file_name, 'w+') 
        # print('p cnf ' + str(int(num_Q_vars + num_D_vars + num_leakage_vars)) + " " + str(int(num_Q_constraints + num_D_constraints + num_leakage_constraints)) + "\n")
        f.write('p cnf ' + str(int(num_Q_vars + num_D_vars + num_leakage_vars)) + " " + str(int(num_Q_constraints + num_D_constraints + num_leakage_constraints)) + "\n") 

        # write the query matrix structure constraints 
        for i in range(self.t_number_of_queries): 
            write_range_clauses_to_file(Q_vars[i], Q_extra_vars[i], f)

        # write the fact that each of the first n needs to be between 0 and r, 
        # write the fact that the others all need to be 0 
        log_r = len(bit_decomposition(self.r_number_of_records))
        R = 2 ** log_r 
        N = 2 ** len(bit_decomposition(self.n_domain_size)) 
        n = self.n_domain_size
        # for the first n files, write that it cannot be greater than n 
        for i in range(n): 
            D_vars_row = D_vars[i]
            for j in range(self.r_number_of_records + 1, R):
                bits = bit_decomposition(j) 
                tmp = []
                for bit in bits: 
                    if bit:
                        tmp.append(-1) 
                    else: 
                        tmp.append(1) 
                # print(len(D_vars_row))
                # print(len(bits))
                assert(len(D_vars_row) == len(bits)) 
                result = [a * b for a, b in zip(D_vars_row, tmp)]
                # f.write(" ".join([str(x) for x in result]) + " 0\n")

        for i in range(n, N): 
            D_vars_row = D_vars[i]
            for x in D_vars_row: 
                f.write(str(-1 * x) + " 0\n") 
        
        write_sum_r_nums_to_n(D_vars, sum_vars[self.t_number_of_queries], carry_vars[self.t_number_of_queries], carry_extra_vars[self.t_number_of_queries], bit_decomposition(self.r_number_of_records), f)

        for i in range(self.t_number_of_queries):
            vec = []
            for j in range(self.n_domain_size):
                tmp_vec = []
                for k in range(log_r):
                    tmp_vec.append(rid_vars[(i,j,k)])
                    # f.write(str(int(rid_vars[(i,k,j,k)])) + " 0\n")
                    f.write(str(int(-1 * rid_vars[(i,j,k)])) + " " + str(int(Q_vars[i][j])) + " 0\n")
                    f.write(str(int(-1 * rid_vars[(i,j,k)])) + " " + str(int(D_vars[j][k])) + " 0\n")
                    f.write(str(int(rid_vars[(i,j,k)])) + " " + str(int(-1 * D_vars[j][k])) + " " + str(int(-1 * Q_vars[i][j])) + " 0\n")
                    # f.write( + str(int(-1 * D_vars[k][j])) + " 0\n")
                vec.append(tmp_vec)
            for j in range(self.n_domain_size,N): 
                tmp_vec = [] 
                for k in range(log_r): 
                    tmp_vec.append(D_vars[j][k])
                vec.append(tmp_vec) 
            write_sum_r_nums_to_n(vec, sum_vars[i], carry_vars[i], carry_extra_vars[i], bit_decomposition(leakage[i]), f)

        f.close()

        with open(self.file_name, 'r+') as f: 
            lines = f.readlines() 
            num_lines = len(lines)
            print('p cnf ' + str(total_vars) + " " + str(num_lines - 1) + "\n")
            lines[0] = 'p cnf ' + str(total_vars) + " " + str(num_lines - 1) + "\n"
        
        with open(self.file_name, 'w') as f: 
            f.writelines(lines)
        
        f.close() 

        # print(f"Elapsed: {(end - start) / (10 ** 9)} s")
        # print("Solving ...")
        cmd = './build/cadical ' +  self.file_name + ' > testing.txt'
        os.system(cmd) 

        output = parse_output_file(D_vars, self.r_number_of_records, self.n_domain_size, num_Q_vars + num_D_vars)

        return output
    
    def solve(self, counts, leakage):  
        start = time.perf_counter_ns() 
        recovered_counts = self.run_iteration(leakage)
        # (recovered_D_matrix)

        end = time.perf_counter_ns() 
        runtime = (end - start) / (10 ** 9)

        print(counts)
        print(recovered_counts)

        recovery_rate = compute_recovery_rates(counts, recovered_counts)
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

    queries_dist = Uniform(number_of_ranges)

    sample = queries_dist.sample(t_number_of_queries)
    random_F = [i for i in range(number_of_ranges)]
    random.shuffle(random_F)

    queries = [random_F[q] for q in sample]

    # data = Uniform(n_domain_size).sample(r_number_of_records)

    Q_matrix = [[0 for _ in range(n_domain_size)] for _ in range(t_number_of_queries)]
    D_matrix = [[0 for _ in range(n_domain_size)] for _ in range(r_number_of_records)]

    for i in range(t_number_of_queries): 
        (lb,ub) = set_of_ranges[queries[i]]
        for j in range(lb,ub+1):
            Q_matrix[i][j] = 1
    counts = [0 for _ in range(n_domain_size)]

    with open('mimic_t4.pkl', 'rb') as f:
        data = pickle.load(f)

    matrix_idx = [np.nonzero(x)[0][0] for x in data]
    for i in range(len(matrix_idx)): 
        D_matrix[i][matrix_idx[i]] = 1
        counts[matrix_idx[i]] += 1

    print(counts)
    # print(Q_matrix)
    # print(data)

    L = np.matmul(Q_matrix, np.array(D_matrix).T.tolist()) 

    # range_rlen_solver = RangeRlenCountSolver(
    #     t_number_of_queries=t_number_of_queries,
    #     r_number_of_records=r_number_of_records,
    #     n_domain_size=n_domain_size,
    #     file_name="test.cnf"
    # )
    recovered_counts = [0 for _ in range(n_domain_size)]
    recovered_data = Uniform(n_domain_size).sample(r_number_of_records)
    for i in range(len(recovered_data)): 
        recovered_counts[recovered_data[i]] += 1
    
    recovery_rate = compute_recovery_rates(counts,recovered_counts)
    time_result = 0
    # print([sum(x) for x in L])
    # time_result, recovery_rate = range_rlen_solver.solve(counts, [sum(x) for x in L])
    return time_result, recovery_rate

def main():
    f = open("range_rlen_count_mimic_uniform.txt", "w")
    t_list = [50,100,150,200,250]
    r_list = [500]
    n_list = [64]
    num_iters = 5
    num_threads = 1

    for t_number_of_queries in t_list: 
        for r_number_of_records in r_list: 
            for n_domain_size in n_list: 
                with Pool(num_threads) as p: 
                    results = p.starmap(run_one_instance, [(t_number_of_queries,r_number_of_records,n_domain_size,) for _ in range(num_iters)])
                    output = np.array(results)
                    for line in output: 
                        print(line)
                        f.write(" ".join([str(t_number_of_queries), str(n_domain_size), str(round(line[0],2)), str(round(line[1],2))]) + " \n")
                    averages = np.array(results).sum(axis=0) / num_iters
                    print(t_number_of_queries, n_domain_size, averages[0], averages[1])

# writes A XOR B = C to CNF
def write_xor_to_file(a,b,c,file): 
    file.write(str(a) + " " + str(b) + " " + str(-1 * c) + " 0\n")
    file.write(str(-1 * a) + " " + str(-1 * b) + " " + str(-1 * c) + " 0\n")
    file.write(str(-1 * a) + " " + str(b) + " " + str(c) + " 0\n")
    file.write(str(a) + " " + str(-1 * b) + " " + str(c) + " 0\n")

# writes A AND B = C to CNF 
def write_and_to_file(a,b,c,file): 
    file.write(str(-1 * a) + " " + str(-1 * b) + " " + str(c) + " 0\n")
    file.write(str(-1 * c) + " " + str(a) + " 0\n")
    file.write(str(-1 * c) + " " + str(b) + " 0\n")

# writes (A AND B) OR (C AND D) = E to file in CNF 
# used for carry bit in the summation 
def write_carry_term_to_file(a,b,c,d,e,file): 
    file.write(str(-1 * a) + " " + str(-1 * b) + " " + str(e) + " 0\n")
    file.write(str(-1 * c) + " " + str(-1 * d) + " " + str(e) + " 0\n")
    file.write(str(-1 * e) + " " + str(a) + " " + str(c) + " 0\n")
    file.write(str(-1 * e) + " " + str(a) + " " + str(d) + " 0\n")
    file.write(str(-1 * e) + " " + str(b) + " " + str(c) + " 0\n")
    file.write(str(-1 * e) + " " + str(b) + " " + str(d) + " 0\n")

# make sure the LSB is on the left for every term 
def sum_two_nums_to_n(num1, num2, sum_bits, carry_bits, carry_extra_bits, file):
    k = len(num1)
    assert(k == len(num2))
    assert(k == len(sum_bits))
    assert(k == len(carry_bits)) 
    assert(len(carry_extra_bits) + 1 == k)
    # assert(len(n_bits) == k + 1)

    # s0 = a0 XOR b0 
    write_xor_to_file(num1[0], num2[0], sum_bits[0], file) 
    # c0 = a0 AND b0 
    write_and_to_file(num1[0], num2[0], carry_bits[0], file) 

    for i in range(1, k): 
        # si = ai XOR bi XOR ci-1 
        # 1. ti-1  = ai xor bi
        write_xor_to_file(num1[i], num2[i], carry_extra_bits[i-1], file)
        # 2. si = ti-1 XOR ci-1 
        write_xor_to_file(carry_extra_bits[i-1], carry_bits[i-1], sum_bits[i], file)
        # carry bit sum 
        write_carry_term_to_file(num1[i], num2[i], carry_bits[i-1], carry_extra_bits[i-1], carry_bits[i], file)
    
    # write the conditions for the sum to file
    # for i in range(len(sum_bits)): 
    #     bit = n_bits[i] 
    #     if bit: 
    #         file.write(str(sum_bits[i]) + " 0\n")
    #     else: 
    #         file.write(str(-1 * sum_bits[i]) + " 0\n")
    # if n_bits[-1]: 
    #     file.write(str(carry_bits[-1]) + " 0\n")
    # else: 
    #     file.write(str(-1 * carry_bits[-1]) + " 0\n")
    sum_bits.append(carry_bits[-1])
    return sum_bits 

# def gen_range_min_cover(range_query, n_domain_size): 
#     output = [0 for _ in range(2*n_domain_size - 1)]

#     for i in range(range_query[0],range_query[1] + 1): 
#         output[i] = 1
#     prev_level_start_index = 0 
#     level_size = n_domain_size // 2

#     while level_size > 0: 
#         for i in range(level_size): 
#             left_node = prev_level_start_index + 2*i 
#             right_node = prev_level_start_index + 2*i + 1 

#             if output[left_node] & output[right_node]: 
#                 output[prev_level_start_index + 2*level_size + i] = 1
#                 output[left_node] = 0
#                 output[right_node] = 0
            
#         prev_level_start_index += 2*level_size 
#         level_size //= 2
#     return output

def write_sum_r_nums_to_n(nums, sum_bits, carry_bits, carry_extra_bits, n_bits, file): 
    assert(len(nums) & (len(nums) - 1) == 0)
    assert(len(nums) != 0)

    curr_round = 0 
    round_size = len(nums) // 2 

    prev_sums = nums.copy() 

    while round_size > 0: 
        prev_sums = nums.copy() 
        nums = []
        for i in range(round_size): 
            num1 = prev_sums[2 * i]
            num2 = prev_sums[2 * i + 1] 
            nums.append(sum_two_nums_to_n(num1,num2,sum_bits[curr_round][i],carry_bits[curr_round][i],carry_extra_bits[curr_round][i], file))
        
        curr_round += 1
        round_size //=2 
    
    log_n = len(n_bits) 
    for i in range(log_n): 
        bit = n_bits[i] 
        if bit: 
            file.write(str(nums[0][i]) + " 0\n")
        else: 
            file.write(str(-1 * nums[0][i]) + " 0\n")
    
    # set everything else to 0
    for i in range(log_n, len(nums[0])): 
        file.write(str(-1 * nums[0][i]) + " 0\n")

# def write_sum_r_nums_to_n(nums, sum_bits, carry_bits, carry_extra_bits, n_bits, file):
#     r = len(nums) 
#     assert(len(sum_bits) == r - 1)
#     assert(len(carry_bits) == r - 1)
#     assert(len(carry_extra_bits) == r - 1)

#     # first sum 
#     running_sum = sum_two_numbers_to_n(nums[0],nums[1], sum_bits[0], carry_bits[0], carry_extra_bits[0], file) 
#     for i in range(2, r): 
#         running_sum = sum_two_numbers_to_n(running_sum, nums[i], sum_bits[i-1], carry_bits[i-1], carry_extra_bits[i-1], file)

#     log_n = len(n_bits) 
#     for i in range(log_n): 
#         bit = n_bits[i] 
#         if bit: 
#             file.write(str(running_sum[i]) + " 0\n")
#         else: 
#             file.write(str(-1 * running_sum[i]) + " 0\n")
    
#     # set everything else to 0 
#     for i in range(log_n, len(running_sum)): 
#         file.write(str(-1 * running_sum[-1]) + " 0\n")

if __name__ == "__main__":
    main()
    # nums = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]]
    # sum_bits = {0: [[21,22,23,24,25],[26,27,28,29,30]], 1: [[31,32,33,34,35,36]]}
    # carry_bits = {0: [[37,38,39,40,41],[42,43,44,45,46]], 1: [[47,48,49,50,51,52]]}
    # carry_extra_bits = {0:[[53,54,55,56],[57,58,59,60]], 1:[[61,62,63,64,65]]}
    # # sum_bits = [[17,18,19,20,21],[22,23,24,25,26,27]]
    # # carry_bits = [[28,29,30,31,32],[33,34,35,36,37,38]]
    # # carry_extra_bits = [[39,40,41,42],[43,44,45,46,47]]
    # n_bits = [1,1,1,1,1,1]
    
    # file = open('test.cnf','w+')

    # write_sum_r_nums_to_n(nums,sum_bits,carry_bits,carry_extra_bits,n_bits,file)

    # bit_decomposition(5)
    # with open('mimic_t4.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # matrix_idx = [np.nonzero(x)[0][0] for x in data]
    # n = max(matrix_idx) + 1
    # queries = Uniform(n).sample(len(matrix_idx))

    # counts = [0 for _ in range(max(matrix_idx) + 1)]
    # recovered_counts =  [0 for _ in range(max(matrix_idx) + 1)]
    # for i in range(len(matrix_idx)): 
    #     #D_matrix[i][matrix_idx[i]] = 1
    #     counts[matrix_idx[i]] += 1
    #     recovered_counts[queries[i]] += 1