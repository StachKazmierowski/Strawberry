import numpy as np
import pandas as pd
from symmetrized.utils import divides, symmetrized_pure_payoff_a
from itertools import permutations
import scipy.special
import math

# Mock data
strategy_one = divides(10,6)[0]
strategy_two = divides(10,6)[1]
print(strategy_one, strategy_two.shape)

def single_payoff_matrix(strategy_A, strategy_B):
    assert strategy_A.shape == strategy_B.shape
    fields_number = strategy_A.shape[0]
    matrix = np.zeros((fields_number, fields_number))
    reversed_strategy_B = strategy_B[::-1]
    for i in range(fields_number):
        for j in range(fields_number):
            matrix[i,j] = reversed_strategy_B[j] - strategy_A[i]
    matrix = np.sign(matrix)
    pd_matrix = pd.DataFrame(matrix, columns=reversed_strategy_B.tolist(), index=strategy_A.tolist())
    print(pd_matrix)
    return matrix

def single_payoff_matrix_vectors(strategy_A, strategy_B):
    matrix = single_payoff_matrix(strategy_A, strategy_B)
    W = []
    T = []
    fields_num = matrix.shape[0]
    for i in range(fields_num):
        tmp_w = 0
        tmp_t = 0
        while(matrix[fields_num - 1 - tmp_w, i] == 1):
            tmp_w += 1
        while(matrix[fields_num - 1 - tmp_t - tmp_w, i] == 0):
            tmp_t += 1
        W.append(tmp_w)
        T.append(tmp_t)
    W = np.array(W)
    T = np.array(T)
    return W, T

print(single_payoff_matrix_vectors(strategy_one, strategy_two))

def payoff(strategy_A, strategy_B):
    matrix = single_payoff_matrix(strategy_A, strategy_B)
    fields_number = matrix.shape[0]
    perm = list(range(fields_number))
    perm = list(set(permutations(perm)))
    fields_number_fac = len(perm)
    winning_A = 0
    ties = 0
    winning_B = 0
    tmp = 0
    for i in range(len(perm)):
        for j in range(len(perm[i])):
            tmp += matrix[j, perm[i][j]]
        if(tmp > 0):
            winning_A += 1
        elif(tmp == 0):
            ties += 1
        else:
            winning_B += 1
        tmp = 0
    assert -(winning_A - winning_B) / fields_number_fac == symmetrized_pure_payoff_a(strategy_A, strategy_B)
    return -(winning_A - winning_B) / fields_number_fac

# print(payoff(strategy_one, strategy_two))

#%%
# strategy_one = np.array([5,3])
# strategy_two = np.array([4,4])

# print(symmetrized_pure_payoff_a(strategy_one, strategy_two))
def newton_symbol(n,k):
    return scipy.special.comb(n , k, exact=True)

def factorial(n):
    if(n < 0):
        return 0
    else:
        return math.factorial(n)

def single_type_rectangle(rows_num, cols_num, rooks_num):
    if(rooks_num > cols_num or rooks_num > rows_num):
        return 0
    if(rooks_num == 0):
        return 1
    return newton_symbol(rows_num, rooks_num) * newton_symbol(cols_num, rooks_num) * factorial(rooks_num)

def max_rook_num(W):
    if(len(W) == 0 or np.max(W) <= 0):
        return 0
    return 1 + max_rook_num(np.delete(W, np.argmax(W)) - 1)

def L_vector(W, T, fields_num):
    return - W - T + fields_num

def prepare_H(strategy_A, strategy_B):
    W, T = single_payoff_matrix_vectors(strategy_A, strategy_B)


print(L_vector(single_payoff_matrix_vectors(strategy_one, strategy_two)[0],
               single_payoff_matrix_vectors(strategy_one, strategy_two)[1], 6))

# how many ways are the to put m rook in (i,j) chessboard with k_W in W and k_L in L, where W, T and L are described
# by W and T
def H(i, j, m, k_W, k_L, W, T):
    return 0


# print(single_payoff_matrix(strategy_one, strategy_two))
# print(single_type_rectangle(5,5,4))
