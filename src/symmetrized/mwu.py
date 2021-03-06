from utils import get_matrix_numpy
import numpy as np
import time
# from solutions_evaluator import possible_payoff_increase_B
np.set_printoptions(suppress=True)

def MWU_game_algorithm(payoff_mat, phi=1/2, steps_number=1000):
    # payoff_mat = get_matrix_numpy(A, B, fields_number)
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
        # print(p_t)
        p_t = p_t/p_t.sum()
        p_t_sum = p_t_sum + p_t
    j_distribution = j_sumed/j_sumed.sum()
    return p_best, j_distribution

# start_time = time.time()
# res = MWU_game_algorithm(10,10,3,1/8,10000)
# print("czas", time.time() - start_time)
# # print(np.matmul(payoff_matrix(6,6,5), res))
# print(res[0].reshape(res[0].shape[1], res[0].shape[0]), "\n", "\n", res[1])

