from symmetrized.solutions_evaluator import eval_strategy, chopstic_row_solution_to_vector, epsilon_value
from symmetrized.mwu import MWU_game_algorithm
from symmetrized.utils import try_reading_matrix_numpy, payoff_matrix
from solutions_parser.chopstic_data_parser import parse_file, parse_game_value
import time
import pandas as pd
import numpy as np
RES_PATH_TIMES = "./results/symmetric/epsilon/times/"
RES_PATH_ERRORS_COL = "./results/symmetric/epsilon/col/"
RES_PATH_ERRORS_ROW = "./results/symmetric/epsilon/row/"

phis_bound = 12
steps_number_bound = 6
phis = [1/2**i for i in range(1, phis_bound)]
phis_names = ["1/2^" + str(i) for i in range(1, phis_bound)]
steps_numbers = [10**i for i in range(1,steps_number_bound)]

def run_epsilon_test(A=7, B=6, fields=5):
    if(A!=B):
        payoff_mat = payoff_matrix(A,B,fields)
    else:
        payoff_mat = try_reading_matrix_numpy(A, B, fields)
    name_part = str(A) + "_" + str(B) + "_" + str(fields)
    epsilons_row = np.zeros((len(phis), len(steps_numbers)))
    epsilons_col = np.zeros((len(phis), len(steps_numbers)))
    times = -np.ones((len(phis), len(steps_numbers)))
    for i in range(len(phis)):
        for j in range(len(steps_numbers)):
            start_time = time.time()
            strategy_A_row, strategy_B_col = MWU_game_algorithm(A, B, fields, phis[i], steps_numbers[j])
            strategy_B_row, strategy_A_col = MWU_game_algorithm(B, A, fields, phis[i], steps_numbers[j])
            # print(strategy_A_col, strategy_B_col.transpose())
            epsilon_col = epsilon_value(strategy_A_row, strategy_B_row.transpose(), -payoff_mat)
            epsilon_row = epsilon_value(strategy_A_col.transpose(), strategy_B_col, payoff_mat)
            # print(epsilon_col)
            # print(epsilon_row)
            if(epsilon_col == 0 and epsilon_row == 0):
                break
            else:
                epsilons_row[i, j] = epsilon_row
                epsilons_col[i, j] = epsilon_col
                times[i, j] = (time.time() - start_time)/2
        if(i > 0 and (np.greater_equal(epsilons_col[i,:], epsilons_col[i-1,:])).all()
                and (np.greater_equal(epsilons_row[i,:], epsilons_row[i-1,:])).all()):
            break
    times = pd.DataFrame(times, index=phis_names, columns=steps_numbers)
    epsilons_row = pd.DataFrame(epsilons_row, index=phis_names, columns=steps_numbers)
    epsilons_col = pd.DataFrame(epsilons_col, index=phis_names, columns=steps_numbers)
    times.to_csv(RES_PATH_TIMES + name_part + ".csv")
    epsilons_row.to_csv(RES_PATH_ERRORS_ROW + name_part + ".csv")
    epsilons_col.to_csv(RES_PATH_ERRORS_COL + name_part + ".csv")

run_epsilon_test(4,4,3)

for res in range(2,11):
    for fields in range(2,11):
        if(res > fields):
            start = time.time()
            run_epsilon_test(res, res, fields)
            print("Liczba zasobów:", res, "Liczba pól:", fields)
            print("Dominika jest hooot, żeby to obliczyć potrzebne było: ", time.time()-start, "sekund")






