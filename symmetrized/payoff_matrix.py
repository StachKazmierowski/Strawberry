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
        T.append(tmp_t + tmp_w)
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

def single_type_rectangle(cols_num, rows_num, rooks_num):
    if(rooks_num > cols_num or rooks_num > rows_num):
        return 0
    if(rooks_num == 0):
        return 1
    return newton_symbol(rows_num, rooks_num) * newton_symbol(cols_num, rooks_num) * factorial(rooks_num)

def max_rook_num(W):
    if(len(W) == 0 or np.max(W) <= 0):
        return 0
    if(len(W) == 1):
        return 1
    multiple_max = (np.argmax(W) == np.argmax(np.delete(W, np.argmax(W))))
    return 1 + max_rook_num(np.delete(W, np.argmax(W)) - multiple_max)
# print(max_rook_num(np.array([1,3,3])))

def L_vector(W, T, fields_num):
    return - T + fields_num

def prepare_H(strategy_A, strategy_B):
    W, T = single_payoff_matrix_vectors(strategy_A, strategy_B)

def width_to_remove(W):
    width = 1
    full_width = W.shape[0]
    for i in range(full_width-1):
        if(W[-i] == W[-1 - i]):
            width += 1
        else:
            break
    return width

# W, T = single_payoff_matrix_vectors(strategy_one, strategy_two)
# print(W[1:-1])
# print(width_to_remove(W[1:-1]))
# how many ways are the to put m rook in (i,j) chessboard with k_W in W and k_L in L, where W, T and L are described
# by W and T

def is_single_type(W, T, j):
    if(np.min(W) == j):
        return True
    return False

def what_single_type(W, T, j):
    if(W[-1] == j):
        return 1 ## type W
    if(T[-1] == j):
        return 0
    return -1

def H_0(i, j, m, k_W, k_L, flag):
    print("Wywołanie H_0")
    print(flag)
    if(m > j or m > i):
        print("H_0, przypadek 1")
        return 0
    if(k_W + k_L > m):
        print("H_0, przypadek 2")
        return 0
    if(flag == 1):
        if(k_L > 0):
            return 0
        if(m != k_W):
            return 0
        print("H_0 zwraca liczbę")
        print(k_W, m)
        return single_type_rectangle(i, j, m)
    if(flag == 0):
        if(k_L > 0):
            return 0
        if(k_W > 0):
            return 0
        return single_type_rectangle(i, j, m)
    if(flag == -1):
        if(k_W > 0):
            return 0
        if(m != k_L):
            return 0
        return single_type_rectangle(i, j, m)


def H(i, j, m, k_W, k_L, W, T):
    print("Wywołanie H")
    if(is_single_type(W, T, j)):
        print("PRZYPADEK 1")
        print("ZWRACAM 0")
        return H_0(i, j, m, k_W, k_L, what_single_type(W, T, j))
    if(m > j or m > i):
        print("PRZYPADEK 2")
        print("ZWRACAM 0")
        return 0
    if(k_W + k_L > m):
        print("PRZYPADEK 3")
        print("ZWRACAM 0")
        return 0
    if(max_rook_num(W) < k_W):
        print("PRZYPADEK 4")
        print("ZWRACAM 0")
        return 0
    if(max_rook_num(L_vector(W, T, j)) < k_L):
        print("PRZYPADEK 5")
        print("ZWRACAM 0")
        return 0
    if(W[-1] == j): ## corner is in W
        print("PRZYPADEK 6")
        width = width_to_remove(W)
        maximum_rooks_in_right = max(width, j)
        sum = 0
        print("width:", width)
        for r in range(min(maximum_rooks_in_right, m, k_W) + 1):
            # print(r)
            sum += H(i - width, j, m - r, k_W - r, k_L, W[:-width], T[:-width]) * single_type_rectangle(width, j - (m - r), r)
        print("ZWRACAM SUMĘ", sum)
        return sum
    if(T[-1] < j): ## corner is in L
        print("PRZYPADEK 7")
        height = j - T[-1]
        maximum_rooks_in_top = max(i, height)
        sum = 0
        print("height:", height)
        for r in range(min(maximum_rooks_in_top, m, k_L) + 1):
            sum += H(i, j - height, m - r, k_W, k_L - r, W , T) * single_type_rectangle(i - (m - r), height, r)
        print("ZWRACAM SUMĘ", sum)
        return sum
    if(T[-1] == j):
        print("PRZYPADEK 8")
        width = width_to_remove(T)
        height = T[-1] - W[-1]
        sum = 0
        print("WESZLIŚMY W REMISY")

    # trzy opcje, róg jest w W, L lub T

print(H(4, 4, 4, 2, 1, np.array([1,1,1,4]), np.array([1,1,1,4])))

# print(width_to_remove(np.array([1,1,1,4])))
# print(single_payoff_matrix(strategy_one, strategy_two))
# print(single_type_rectangle(5,5,4))

