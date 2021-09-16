import numpy as np
import pandas as pd
from itertools import permutations

def next_divide(divide):
    n = divide.shape[1]
    div_num = divide.shape[0]
    dev_tmp = np.empty((0, n), int)
    for i in range(div_num):
        tmp = divide[i][:]
        for j in range(n):
            if (j == 0 or tmp[j] < tmp[j - 1]):
                tmp[j] = tmp[j] + 1
                dev_tmp = np.append(dev_tmp, tmp.reshape(1, n), axis=0)
                tmp[j] = tmp[j] - 1
    return (np.unique(dev_tmp, axis=0))

def divides(A, n):
    if (A == 0):
        return np.zeros((1, n))
    devs = np.zeros((1, n))
    devs[0][0] = 1
    for i in range(A - 1):
        devs_next = next_divide(devs)
        devs = devs_next
    return (devs)

def symmetrized_more_than_opponet_payofff(x_a, x_b):
  tmp_b = np.array(list(set(permutations(x_b.tolist()))))
  signum = (np.sign((tmp_b - x_a)).sum(axis=1))
  pure_payoffs = np.sign(signum)
  payoff = pure_payoffs.sum() / tmp_b.shape[0]
  return - payoff

def more_than_half_payofff(A, B):
    fields_num = A.shape[0]
    # print(A)
    # print(A.shape)
    signum = np.sign(A - B)
    A_wins = (signum > 0).sum()
    B_wins = (signum < 0).sum()
    if(A_wins > fields_num / 2):
        return 1
    if(B_wins > fields_num / 2):
        return -1
    return 0

def symmetrized_more_than_half_payoff(A, B):
    result = 0
    B_permutations = np.array(list(set(permutations(B.tolist()))))
    for i in range(B_permutations.shape[0]):
        result += more_than_half_payofff(A, B_permutations[i])
    return result / B_permutations.shape[0]

def payoff_matrix_more_than_half(A, B, n):
    A_strategies = divides(A, n)
    B_strategies = divides(B, n)
    matrix = np.zeros((A_strategies.shape[0], B_strategies.shape[0]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i ,j] = symmetrized_more_than_half_payoff(A_strategies[i], B_strategies[j])
    return matrix

def payoff_matrix_more_than_opponent(A, B, n):
    A_strategies = divides(A, n)
    B_strategies = divides(B, n)
    matrix = np.zeros((A_strategies.shape[0], B_strategies.shape[0]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i ,j] = symmetrized_more_than_opponet_payofff(A_strategies[i], B_strategies[j])
    return -matrix

def payoff_matrix_more_than_half_pd(A, B, n):
    matrix = payoff_matrix_more_than_half(A, B, n)
    A_symmetrized_strategies = divides(A, n)
    B_symmetrized_strategies = divides(B, n)
    columns_names = []
    rows_names = []
    for i in range(A_symmetrized_strategies.shape[0]):
        rows_names.append(str(A_symmetrized_strategies[i]))
    for i in range(B_symmetrized_strategies.shape[0]):
        columns_names.append(str(B_symmetrized_strategies[i]))
    df = pd.DataFrame(matrix, columns=columns_names, index=rows_names)
    return df

def payoff_matrix_more_than_opponent_pd(A, B, n):
    matrix = payoff_matrix_more_than_opponent(A, B, n)
    A_symmetrized_strategies = divides(A, n)
    B_symmetrized_strategies = divides(B, n)
    columns_names = []
    rows_names = []
    for i in range(A_symmetrized_strategies.shape[0]):
        rows_names.append(str(A_symmetrized_strategies[i]))
    for i in range(B_symmetrized_strategies.shape[0]):
        columns_names.append(str(B_symmetrized_strategies[i]))
    df = pd.DataFrame(matrix, columns=columns_names, index=rows_names)
    return df

# print(payoff_matrix_more_than_half_pd(10, 10, 4))
# print(payoff_matrix_more_than_opponent_pd(10, 10, 4))


