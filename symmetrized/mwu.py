from utils import payoff_matrix, pd_payoff_matrix, divides
import numpy as np
import pandas as pd
import time
np.set_printoptions(suppress=True)

def MWU_symmetric_game_algorithm(resources_number, fields_number, phi=1/2, steps_number=1000):
    try:
        path = "./data/payoff_matrices/payoff_matrix(" + str(resources_number) + \
               "," + str(resources_number) + "," + str(fields_number) + ").csv"
        payoff_mat = np.delete(pd.read_csv(path).to_numpy(), 0,1)
    except:
        print("Loaded failed")
        payoff_mat = payoff_matrix(resources_number, resources_number, fields_number)
    payoff_mat *= -1
    # print(payoff_mat)
    rows_number = payoff_mat.shape[0]
    cols_number = payoff_mat.shape[1]

    p_0 = np.ones((1, rows_number))
    p_0 = p_0/rows_number
    p_t = p_0
    j_sumed = np.zeros((cols_number, 1))
    smallest_column_payoff = 1
    t_smallest_column_payoff = 0
    p_best = p_0
    p_t_sum = np.zeros((1, rows_number))
    for i in range (steps_number):
        payoffs = np.matmul(p_t, payoff_mat)
        j_best_response = np.argmax(payoffs)
        if(payoffs[0, j_best_response] < smallest_column_payoff):
            t_smallest_column_payoff = i
            smallest_column_payoff = payoffs[0, j_best_response]
            p_best = p_t
        j_sumed[j_best_response] += 1
        m_t = payoff_mat[:,j_best_response]
        p_t = np.multiply((1 - phi * m_t), p_t)
        p_t = p_t/p_t.sum()
        p_t_sum = p_t_sum + p_t
    print(smallest_column_payoff) # lambda + epsilon
    j_distribution = j_sumed/j_sumed.sum()
    print(np.matmul(payoff_mat, j_distribution).min()) # lambda - epsilon
    game_value = smallest_column_payoff + np.matmul(payoff_mat, j_distribution).min()
    print(game_value)
    return p_best, j_distribution
start_time = time.time()
res = MWU_symmetric_game_algorithm(6,5,1/2,500000)
print("czas", time.time() - start_time)
# print(np.matmul(payoff_matrix(6,6,5), res))
# print(res[0].reshape(res[0].shape[1], res[0].shape[0]), "\n", "\n", res[1])
