from src.payoff_matrix_finding.payoff_matrix import permutations_results, single_payoff_matrix_vectors, H
from src.symmetrized.utils import divides
import numpy as np

def payoff_from_perms_result(strategy_one, strategy_two):
    res = permutations_results(strategy_one, strategy_two)
    W = 0
    L = 0
    for i in range(res.shape[0]):
        for j in range(res.shape[0]):
            if(i > j):
                L += res[i,j]
            if(j > i):
                W += res[i,j]
    return (W - L) / res.sum()

def H_results(strategy_one, strategy_two):
    fields_num = strategy_one.shape[0]
    W, T = single_payoff_matrix_vectors(strategy_one, strategy_two)
    res = np.zeros((fields_num + 1, fields_num + 1))
    for i in range(fields_num + 1):
        for j in range(fields_num + 1):
            res[i,j] = H(fields_num, fields_num, fields_num, i, j, W, T)
    return res

def H_diffs(strategy_one, strategy_two):
    mat = H_results(strategy_one, strategy_two)
    fields_num = strategy_one.shape[0]
    wins = np.zeros((fields_num, 1))
    loses = np.zeros((fields_num, 1))
    ties = 0
    for x in range (fields_num + 1):
        wins_tmp = 0
        loses_tmp = 0
        for i in range(fields_num + 1):
            for j in range(fields_num + 1):
                if(x == 0 and i == j):
                    ties += mat[i, j]
                else:
                    if(i - j == x):
                        wins_tmp += mat[i, j]
                    if(j - i == x):
                        loses_tmp += mat[i, j]
        if(x > 0):
            wins[x-1, 0] = wins_tmp
            loses[x-1, 0] = loses_tmp
    return wins, loses, ties
#%%
def test(ties, print_errors):
    A_divides = divides(15, 6)
    B_divides = divides(16, 6)
    num_errors = 0
    num_tries = 0
    error_A = []
    error_B = []
    for A in A_divides:
        for B in B_divides:
            if(ties or np.unique(A).shape[0] + np.unique(B).shape[0] == np.unique(np.append(A, B, axis=0)).shape[0]):
                num_tries += 1
                diff = H_results(A, B) - permutations_results(A, B)
                if(np.max(diff) != 0 or np.min(diff) != 0):
                    print("BŁĄD dla ", A, B)
                    # print(diff)
                    num_errors += 1
                    error_A.append(A)
                    error_B.append(B)
                else:
                    print("OK dla ", A, B)
    print("Liczba błędów", num_errors)
    print("Liczba prób", num_tries)
    if(print_errors):
        for i in range(len(error_A)):
            print(error_A[i], error_B[i])

# test(True, False)
#%%
# A = np.array([5,4,2])
# B = np.array([4,3,3])
# print(permutations_results(A, B))
# print(H_results(A, B))
# # print(H(2,2,2,1,1,np.array([1,1]),np.array([1,1])))
# print(H_diffs(A, B))
