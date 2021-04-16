from symmetrized.payoff_matrix import permutations_results, single_payoff_matrix, \
    single_payoff_matrix_vectors, max_rook_num, H
from symmetrized.utils import symmetrized_pure_payoff_a, divides
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
    # print(fields_num)
    W, T = single_payoff_matrix_vectors(strategy_one, strategy_two)
    # print(W)
    res = np.zeros((fields_num, fields_num))
    for i in range(fields_num):
        for j in range(fields_num):
            res[i,j] = H(fields_num, fields_num, fields_num, i, j, W, T)
    return res
A = np.array([4,4,1,1])
B = np.array([3,3,2,2])
# print(H_results(A, B))
# print(permutations_results(A, B))
#%%
A = np.array([2,1,1])
B = np.array([4,0,0])
print(H_results(A,B))

#%%
A_divides = divides(11, 4)
B_divides = divides(10, 4)

for A in A_divides:
    for B in B_divides:
        if(np.unique(A).shape[0] + np.unique(B).shape[0] == np.unique(np.append(A, B, axis=0)).shape[0]):
            diff = H_results(A, B) - permutations_results(A, B)
            if(np.max(diff) != 0 or np.min(diff) != 0):
                print("BŁĄD dla ", A, B)
                # print(diff)
            else:
                print("OK dla ", A, B)
## testowanie max_rook_num
# def test_max_rook_num(startegy_one, strategy_two):



# print(payoff_from_perms_result(A, B))
# print(symmetrized_pure_payoff_a(A, B))
