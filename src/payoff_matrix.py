import numpy as np
import scipy.special
import math

def single_payoff_matrix(strategy_A, strategy_B):
    assert strategy_A.shape == strategy_B.shape
    fields_number = strategy_A.shape[0]
    matrix = np.zeros((fields_number, fields_number))
    reversed_strategy_B = strategy_B[::-1]
    for i in range(fields_number):
        for j in range(fields_number):
            matrix[i,j] = reversed_strategy_B[j] - strategy_A[i]
    matrix = np.sign(matrix)
    return matrix

def single_payoff_matrix_vectors(strategy_A, strategy_B):
    matrix = single_payoff_matrix(strategy_A, strategy_B)
    W = []
    T = []
    fields_num = matrix.shape[0]
    for i in range(fields_num):
        tmp_w = 0
        tmp_t = 0
        while(tmp_w < fields_num and matrix[fields_num - 1 - tmp_w, i] == 1):
            tmp_w += 1
        while(tmp_w + tmp_t < fields_num and matrix[fields_num - 1 - tmp_t - tmp_w, i] == 0):
            tmp_t += 1
        W.append(tmp_w)
        T.append(tmp_t + tmp_w)
    W = np.array(W)
    T = np.array(T)
    return W, T

def newton_symbol(n,k):
    return scipy.special.comb(n , k, exact=True)

def factorial(n):
    if(n < 0):
        return 0
    else:
        return math.factorial(n)

def single_type_rectangle(cols_num, rows_num, rooks_num):
    if(cols_num < 0 or rows_num < 0 or rooks_num < 0):
        return 0
    if(rooks_num > cols_num or rooks_num > rows_num):
        return 0
    if(rooks_num == 0):
        return 1
    return newton_symbol(rows_num, rooks_num) * newton_symbol(cols_num, rooks_num) * factorial(rooks_num)

def max_rook_num(W):
    tmp_W = []
    for i in range(W.shape[0]):
        if(W[i] > 0):
            tmp_W.append(W[i])
    W = np.array(tmp_W)
    if (len(W) == 0):
        return 0
    if (len(W) == 1):
        return 1
    if (np.all(W - 1 > 0)):
        W = np.delete(W, W.argmin())
    return 1 + max_rook_num(W - 1)

def L_vector(W, T, fields_num):
    return - T + fields_num

def width_to_remove(W):
    width = 1
    full_width = W.shape[0]
    for i in range(full_width-1):
        if(W[-i - 1] == W[-i - 2]):
            width += 1
        else:
            break
    return width

def is_single_type(W, T, j):
    if(np.min(W) == j):
        return True
    if(np.min(T) == j and np.max(W) == 0):
        return True
    if(np.max(T) == 0):
        return True
    return False

def what_single_type(W, T, j):
    if(W[-1] == j):
        return 1 ## type W
    if(T[-1] == j):
        return 0
    return -1

def is_double_type_with_tie(W, T, j):
    if(np.min(W) > 0 and np.min(W) == np.max(W) and np.min(T) == np.max(T) and T[-1] - W[-1] > 0): # T and W
        return True
    if(np.max(W) == 0 and np.max(T) > np.min(T)): # L and T
        return True
    return False

def what_double_type(W, T, j):
    if(np.max(W) == 0):
        return -1
    else:
        return 1

def vector_min(A, k):
    B = np.array(A)
    for i in range(A.shape[0]):
        if(B[i] > k):
            B[i] = k
    return B