import math
import numpy as np
import pandas as pd
from utils import try_reading_matrix_numpy
from solutions_evaluator import epsilon_value
import time
np.set_printoptions(suppress=True)
from mwu import MWU_game_algorithm as MWU_basic
# from mwu_without_time_hoop import SIGNIFICANCE_CONST
SIGNIFICANCE_CONST = 10**(-20)
import sys
EXPERIMENTS_RESULTS_PATH = "../experiments_results_small_support**2/"

def MWU_game_algorithm(payoff_mat, phi=1/2, steps_number=10000):
    rows_number = payoff_mat.shape[0]
    cols_number = payoff_mat.shape[1]
    indexes = get_sorted_indexes(best_pure_responses_indexes(payoff_mat, np.ones((cols_number, 1))))
    p_t = np.ones((1, len(indexes)))
    p_t = normalize(p_t)
    j_sumed = np.zeros((cols_number, 1))
    smallest_column_payoff = 1
    p_best = p_t
    best_indexes = indexes
    start = time.time()
    # p_t_sum = np.zeros((1, rows_number))
    k = 1
    tmp_mat = payoff_mat[indexes,:]
    for i in range (1, steps_number):
        payoffs = np.matmul(p_t, tmp_mat)
        j_best_response = np.argmax(payoffs)
        if(payoffs[0, j_best_response] < smallest_column_payoff):
            smallest_column_payoff = payoffs[0, j_best_response]
            p_best = p_t
            best_indexes = indexes
        j_sumed[j_best_response] += 1
        m_t = tmp_mat[:,j_best_response]
        # p_t = np.multiply((1 - phi * m_t), p_t)
        # m_t = payoff_mat[:,j_best_response]

        m_t_negative = (m_t < 0)
        p_t_significant = (p_t > SIGNIFICANCE_CONST)
        to_update = np.logical_or(m_t_negative, p_t_significant[0])
        m_t_updating = np.where(to_update,m_t,0)
        p_t_updating = np.where(to_update,p_t,0)
        p_t = np.multiply((1 - phi * m_t_updating), p_t_updating)


        # p_t_sum = p_t_sum + p_t
        j_distribution = j_sumed/j_sumed.sum()
        if(i == k and i < steps_number/2):
            print(i)
            new_indexes = get_sorted_indexes(best_pure_responses_indexes(payoff_mat, j_distribution))
            p_t = update_p_t(p_t, indexes, new_indexes, 1/rows_number)
            indexes = new_indexes
            # for index in range(len(to_ressurect)):
            #     if(p_t[0][to_ressurect[index]] == 0):
            #         p_t[0][to_ressurect[index][0]] = 1 / p_t.shape[1]
            #         print("updating index: " + str(to_ressurect[index][0]) + ", with value: ", 1 / p_t.shape[1])
            k += 2**int(math.log2(i))
            tmp_mat = payoff_mat[indexes,:]
        p_t = p_t/p_t.sum()
    p_best = translate_p_t(p_best, best_indexes, rows_number)
    game_value = np.matmul(np.matmul(p_best, payoff_mat), j_distribution)[0][0]
    row_row = max(epsilon_value(p_best, np.transpose(p_best), payoff_mat))
    col_col = max(epsilon_value(np.transpose(j_distribution), j_distribution, payoff_mat))
    row_col = (max(epsilon_value(p_best, j_distribution, payoff_mat)))
    duration = time.time() - start
    return duration, row_row, row_col, col_col

def translate_p_t(p_t, indexes, rows_num):
    result = np.zeros((1, rows_num))
    for i in range(len(indexes)):
        result[0, indexes[i]] = p_t[0, i]
    return result

def update_p_t(p_t, old_indexes, new_indexes, update_value):
    p_t_new = np.zeros_like(p_t)
    for i in range(len(new_indexes)):
        if new_indexes[i] in old_indexes:
            index = old_indexes.index(new_indexes[i])
            p_t_new[0, i] = p_t[0, index]
    p_t_new[p_t_new == 0] = update_value
    return p_t_new

def get_sorted_indexes(indexes):
    list_indexes = list(indexes[:,0])
    list_indexes.sort()
    return list_indexes

def best_pure_responses_indexes(matrix, column_strat):
    row_payoffs = -np.matmul(matrix, column_strat)
    indexes = np.argsort(row_payoffs, axis=0)
    return indexes[- poly_size(matrix.shape[0]):]

def worst_pure_responses_indexes(matrix, column_strat):
    row_payoffs = -np.matmul(matrix, column_strat)
    indexes = np.argsort(row_payoffs, axis=0)
    return indexes[ : poly_size(matrix.shape[0])]

def normalize(vector):
    return vector/vector.sum()

def start_vector(matrix, rows_number, columns_number):
    p_0 = np.zeros((1, rows_number))
    uniform = np.ones((columns_number, 1))
    uniform = normalize(uniform)
    best = worst_pure_responses_indexes(matrix, uniform)
    p_0[0][best] = 1
    p_0 = normalize(p_0)
    return p_0

def poly_size(number):
    return int(math.log(number) ** 2)

def run_experiment(A, B, n, phis_bound, max_steps_power_mult):
    max_steps = 125 * 2 ** max_steps_power_mult
    matrix = -try_reading_matrix_numpy(A,B,n)

    times = []
    row_row = []
    col_col = []
    row_col = []

    phis = [(1/2)**i for i in range(1, phis_bound+1)]
    columns_names = [125 * 2**i for i in range(max_steps_power_mult + 1)]
    for phi in phis:
        times_tmp = []
        row_row_tmp = []
        col_col_tmp = []
        row_col_tmp = []
        for column_number in columns_names:
            result = MWU_game_algorithm(matrix, phi, column_number)
            times_tmp.append(result[0])
            row_row_tmp.append(result[1])
            col_col_tmp.append(result[2])
            row_col_tmp.append(result[3])
        times.append(times_tmp)
        row_row.append(row_row_tmp)
        col_col.append(col_col_tmp)
        row_col.append(row_col_tmp)
    save_experiments_results(pd.DataFrame(np.array(times), index=phis, columns=columns_names), gen_name(A, B, n), EXPERIMENTS_RESULTS_PATH + "/times/")
    save_experiments_results(pd.DataFrame(np.array(row_row), index=phis, columns=columns_names), gen_name(A, B, n), EXPERIMENTS_RESULTS_PATH + "/row_row/")
    save_experiments_results(pd.DataFrame(np.array(col_col), index=phis, columns=columns_names), gen_name(A, B, n), EXPERIMENTS_RESULTS_PATH + "/col_col/")
    save_experiments_results(pd.DataFrame(np.array(row_col), index=phis, columns=columns_names), gen_name(A, B, n), EXPERIMENTS_RESULTS_PATH + "/row_col/")

def save_experiments_results(data, name, folder=EXPERIMENTS_RESULTS_PATH):
    data.to_csv(folder + name)

def gen_name(A, B, n):
    return "(" + str(A) + "," + str(B) + "," + str(n) + ").csv"

if __name__ == "__main__":
    args = sys.argv
    A,n,phi,max_steps_power_mult = int(args[1]), int(args[2]), int(args[3]), int(args[4])
    run_experiment(A, A, n, phi, max_steps_power_mult)

# A, B, n = 30, 30, 15
# num_steps = 1024000
# phi = (1/2)**6
# matrix = -try_reading_matrix_numpy(A, B, n)
# start = time.time()
# print(best_pure_responses_indexes(matrix, normalize(np.ones((matrix.shape[0], 1)))))
# print(worst_pure_responses_indexes(matrix, normalize(np.ones((matrix.shape[0], 1)))))
# print(start_vector(matrix, matrix.shape[0], matrix.shape[1]))
# result = MWU_game_algorithm(matrix, phi, num_steps)
#
# print(time.time() - start)
# print(epsilon_value(result[0], result[1], matrix))

# print(poly_size(matrix.shape[0]))
