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

print(payoff(strategy_one, strategy_two))

#%%
strategy_one = np.array([5,3])
strategy_two = np.array([4,4])

# print(symmetrized_pure_payoff_a(strategy_one, strategy_two))

## in how many ways can we fill (1, i)x(1, w_i) rectangle, in such way that exactly j rooks are in W = (w_1, w_2, ...,w_n)
def prepare_F(matrix):
    fields_number = matrix.shape[0]
    W = list((matrix > 0).sum(axis=0))
    M = [min(i+1, W[i]) for i in range(fields_number)]
    print(W[0])
    return W, M

def run_F(matrix, j):
    W, M = prepare_F(matrix)
    print(W)
    print(M)
    fields_number = matrix.shape[0]
    print((fields_number - W[-1]))
    return math.factorial(fields_number - W[-1]) * F(fields_number, j, W, M)

def newton_symbol(n,k):
    return scipy.special.comb(n , k, exact=True)

def factorial(n):
    if(n < 0):
        return 0
    else:
        return math.factorial(n)

def F(i, j, W, M):
    if(j > M[i-1]):
        return 0
    if(j == 0):
        if(i == 0):
            return 1
        else:
            return 0
    if(i == 0):
        if(j == 1):
            return W[0]
        else:
            return 0
    m_1 = M[i-2]
    m_2 = M[i-1]
    w_i = [W[i-1], W[i-2]]
    print("Współczynniki")
    first_coef = (newton_symbol((i-1) - m_1, m_2 - m_1) * newton_symbol(w_i[0] - w_i[-1], m_2 - m_1) * factorial(m_2 - m_1))
    second_coef = factorial(m_2 - m_1 - 1) * newton_symbol((i - 1) - m_1, m_2 - m_1 - 1) * (
        (w_i[-1] - m_1) * newton_symbol(w_i[0] - w_i[-1], m_2 - m_1 - 1) +
        + (w_i[0] - w_i[-1]) * newton_symbol(w_i[0] - w_i[-1] - 1, m_2 - m_1 - 1)
    )
    return F(i-1, j, W, M) * first_coef + F(i-1, j-1, W, M) * second_coef


# TODO CZEMU WYCHODZI LISTA???? HEHEHE
print(run_F(single_payoff_matrix(strategy_one, strategy_two), 1))
print(newton_symbol(0, 0))
# print(math.factorial(-1))

