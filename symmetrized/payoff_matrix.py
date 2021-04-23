import numpy as np
import pandas as pd
from symmetrized.utils import divides, symmetrized_pure_payoff_a
from itertools import permutations
import scipy.special
import math

# Mock data
strategy_one = divides(10,6)[0]
strategy_two = divides(10,6)[1]

def single_payoff_matrix(strategy_A, strategy_B):
    assert strategy_A.shape == strategy_B.shape
    fields_number = strategy_A.shape[0]
    matrix = np.zeros((fields_number, fields_number))
    reversed_strategy_B = strategy_B[::-1]
    for i in range(fields_number):
        for j in range(fields_number):
            matrix[i,j] = reversed_strategy_B[j] - strategy_A[i]
    matrix = np.sign(matrix)
    # pd_matrix = pd.DataFrame(matrix, columns=reversed_strategy_B.tolist(), index=strategy_A.tolist())
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

def permutations_results(strategy_A, strategy_B):
    matrix = single_payoff_matrix(strategy_A, strategy_B)
    res = np.zeros((strategy_A.shape[0] + 1, strategy_B.shape[0] + 1))
    fields_number = matrix.shape[0]
    perm = list(range(fields_number))
    perm = list(set(permutations(perm)))
    k_W = 0
    k_L = 0
    for i in range(len(perm)):
        for j in range(len(perm[i])):
            if(matrix[j, perm[i][j]] == 1):
                k_W += 1
            if(matrix[j, perm[i][j]] == -1):
                k_L += 1
        res[k_W, k_L] += 1
        k_W = 0
        k_L = 0
    return res

def newton_symbol(n,k):
    return scipy.special.comb(n , k, exact=True)

def factorial(n):
    if(n < 0):
        return 0
    else:
        return math.factorial(n)

def single_type_rectangle(cols_num, rows_num, rooks_num):
    if(cols_num < 0 or rows_num < 0 or rooks_num < 0):
        # print("UJEMNA WARTOŚĆ W PROSTOKĄCIE")
    #     print(cols_num, rows_num, rooks_num)
        return 0
        ### TODO błędy przy warunku if(cols_num==0) ret 0 - może być tak, że wyrzucamy z prostokąta wszystkie wiersze i nie możemy nic w nim umieścić
    if(rooks_num > cols_num or rooks_num > rows_num):
        return 0
    if(rooks_num == 0):
        return 1
    return newton_symbol(rows_num, rooks_num) * newton_symbol(cols_num, rooks_num) * factorial(rooks_num)

def max_rook_num(W): ## TODO zajebiśćie zrobić tą funkcje bo tu wychodzi większość problemów
    W = np.delete(W, W <= 0)
    if(len(W) == 0):
        return 0
    if(len(W) == 1):
        return 1
    return 1 + max_rook_num(W - 1)

def L_vector(W, T, fields_num):
    return - T + fields_num

def prepare_H(strategy_A, strategy_B):
    W, T = single_payoff_matrix_vectors(strategy_A, strategy_B)

def width_to_remove(W):
    width = 1
    full_width = W.shape[0]
    for i in range(full_width-1):
        if(W[-i - 1] == W[-i - 2]):
            width += 1
        else:
            break
    return width
W = np.array([0,3,3])
print(W[-2])
print(width_to_remove(W))

#%%
def is_single_type(W, T, j):
    if(np.min(W) == j):
        return True
    if(np.min(T) == j and np.max(W) == 0):
        return True
    if(np.max(T) == 0):
        return True
    return False

def what_single_type(W, T, j):
    # print(W, T, j)
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
    for i in range(A.shape[0]):
        if(A[i] > k):
            A[i] = k
    return A

def H_0(i, j, m, k_W, k_L, flag):
    # print("H0")
    # print(flag)
    if(flag == 1):
        if(k_L > 0):
            return 0
        if(m != k_W):
            return 0
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

def H_1(i, j, m, k_W, k_L, W, T, flag):
    if(min(k_L, k_W) != 0 ):
        return 0
    if(max(k_L, k_W) > m):
        return 0
    if(m > min(i,j)):
        return 0
    if(flag == -1):
        width = width_to_remove(T)
        return single_type_rectangle(i - width, j, k_L) * single_type_rectangle(width, j - k_L, m - k_L)
    if(flag == 1):
        height = T[-1] - W[-1]
        return single_type_rectangle(i, j - height, k_W) * single_type_rectangle(i - k_W, height, m - k_W)


def H(i, j, m, k_W, k_L, W, T):
    if(i == 0 or j == 0):
        return 0
    if(m > j or m > i):
        return 0
    if(k_W + k_L > m):
        return 0
    if(max_rook_num(W) < k_W):
        return 0
    if(max_rook_num(L_vector(W, T, j)) < k_L):
        return 0
    if(is_single_type(W, T, j)):
        return H_0(i, j, m, k_W, k_L, what_single_type(W, T, j))
    if(W[-1] == j): ## corner is in W
        # print("W CORNER")
        width = width_to_remove(W)
        maximum_rooks_in_right = min(width, j)
        sum = 0
        for r in range(min(maximum_rooks_in_right, m, k_W) + 1):
            sum += H(i - width, j, m - r, k_W - r, k_L, W[:-width], T[:-width]) * single_type_rectangle(width, j - (m - r), r)
        return sum
    if(T[-1] < j): ## corner is in L
        # print("L CORNER")
        height = j - T[-1]
        maximum_rooks_in_top = min(i, height)
        sum = 0
        for r in range(min(maximum_rooks_in_top, m, k_L) + 1):
            H_tmp = H(i, j - height, m - r, k_W, k_L - r, vector_min(W, j - height) , vector_min(T, j - height))
            top = single_type_rectangle(i - (m - r), height, r)
            sum += H_tmp * top
            ## TODO czy nie powinniśmy zmniejszać W i T? done, bez zmian
        return sum
    # REMISY
    if(is_double_type_with_tie(W, T, j)):
        return H_1(i, j, m, k_W, k_L, W, T, what_double_type(W, T, j))
    if(T[-1] == j): ## corner is in T
        # print("WESZLIŚMY W REMISY")
        width = width_to_remove(T)
        height = T[-1] - W[-1]
        maximum_rooks_in_top = min(i - width, height)
        maximum_rooks_in_right = min(width, j - height)
        maximum_rooks_in_corner = min(width, height)
        sum = 0
        for r_3 in range(min(maximum_rooks_in_top, m, k_L) + 1):
            for r_2 in range(min(maximum_rooks_in_corner, m) + 1):
                for r_1 in range(min(maximum_rooks_in_right, m, k_W) + 1):
                    if(r_1 + r_2 + r_3 <= m):
                        r_4 = m - r_1 - r_2 - r_3
                        H_tmp = H(i - width, j - height, r_4, k_W - r_1, k_L - r_3, vector_min(W[:-width], j - height)
                                  , vector_min(T[:-width], j - height))
                        top = single_type_rectangle(i - width - r_4, height, r_3)
                        corner = single_type_rectangle(width, height - r_3, r_2)
                        right = single_type_rectangle(width - r_2, j - height - r_4, r_1)
                        sum += H_tmp * top * corner * right
            ## TODO czy nie powinniśmy zmniejszać W i T? done, bez zmian
        return sum
        # print("WESZLIŚMY W REMISY")



    # trzy opcje, róg jest w W, L lub T
#%%
print(list(range(1,2)))
#%%
print(H(5, 5, 5, 1, 4, np.array([1,1,4,4]), np.array([2,2,3,3])))
#%%
A = np.array([2,3,4])
B = np.array([2,4,3])
# print(min(A, B))

print(vector_min(A, 3))