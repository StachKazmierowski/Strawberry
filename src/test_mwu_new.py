import math
import numpy as np
from utils import try_reading_matrix_numpy
from solutions_evaluator import epsilon_value
import time
np.set_printoptions(suppress=True)
from mwu import MWU_game_algorithm as MWU_basic

def MWU_game_algorithm(payoff_mat, phi=1/2, steps_number=10000):
    rows_number = payoff_mat.shape[0]
    cols_number = payoff_mat.shape[1]
    p_t = start_vector(payoff_mat, rows_number, cols_number)
    j_sumed = np.zeros((cols_number, 1))
    smallest_column_payoff = 1
    p_best = p_t
    p_t_sum = np.zeros((1, rows_number))
    k = 1
    # print(multiplier)
    for i in range (1, steps_number):
        payoffs = np.matmul(p_t, payoff_mat)
        j_best_response = np.argmax(payoffs)
        if(payoffs[0, j_best_response] < smallest_column_payoff):
            smallest_column_payoff = payoffs[0, j_best_response]
            p_best = p_t
        j_sumed[j_best_response] += 1
        m_t = payoff_mat[:,j_best_response]
        p_t = np.multiply((1 - phi * m_t), p_t)
        p_t_sum = p_t_sum + p_t
        j_distribution = j_sumed/j_sumed.sum()
        if(i == k):
            print(i)
            to_ressurect = best_pure_responses_indexes(payoff_mat, j_distribution)
            for index in range(len(to_ressurect)):
                if(p_t[0][to_ressurect[index]] == 0):
                    p_t[0][to_ressurect[index][0]] = 1 / p_t.shape[1]
                    # print("updating index: " + str(to_ressurect[index][0]) + ", with value: ", 1 / p_t.shape[1])
            k += 2**int(math.log2(i))
        p_t = p_t/p_t.sum()
    game_value = np.matmul(np.matmul(p_best, payoff_mat), j_distribution)[0][0]
    return p_best, j_distribution, -game_value, game_value


def best_pure_responses_indexes(matrix, column_strat):
    row_payoffs = -np.matmul(matrix, column_strat)
    indexes = np.argsort(row_payoffs, axis=0)
    return indexes[-int((math.log(matrix.shape[0]))):]

def normalize(vector):
    return vector/vector.sum()

def start_vector(matrix, rows_number, columns_number):
    p_0 = np.zeros((1, rows_number))
    uniform = np.ones((columns_number, 1))
    uniform = normalize(uniform)
    best = best_pure_responses_indexes(matrix, uniform)
    p_0[0][best] = 1
    p_0 = normalize(p_0)
    return p_0

A, B, n = 10,10,5
num_steps = 1000
phi = (1/2)**5
matrix = -try_reading_matrix_numpy(A, B, n)
start = time.time()
result = MWU_game_algorithm(matrix, phi, num_steps)
print(time.time() - start)
print(epsilon_value(result[0], result[1], matrix))

start = time.time()
result = MWU_basic(matrix, phi, num_steps)
print(time.time() - start)
print(epsilon_value(result[0], result[1], matrix))
