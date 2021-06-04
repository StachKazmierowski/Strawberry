from solutions_evaluator import epsilon_value
from mwu import MWU_game_algorithm
from utils import get_matrix_numpy
import os
import time
import pandas as pd
import numpy as np
RES_PATH_TIMES = "./results/symmetric/times/"
RES_PATH_ERRORS_COL = "./results/symmetric/col/"
RES_PATH_ERRORS_ROW = "./results/symmetric/row/"

phis_bound = 10
steps_number_bound = 20
phis = [1/2**i for i in range(1, phis_bound)]
phis_names = ["1/2^" + str(i) for i in range(1, phis_bound)]
steps_numbers = [2**i for i in range(1,steps_number_bound)]

def run_epsilon_test(A=7, B=6, fields=5):
    name_part = str(A) + "_" + str(B) + "_" + str(fields)
    if(os.path.exists(RES_PATH_TIMES + name_part + ".csv")):
        print("already tested for for:")
        print("Liczba pól", fields, "liczba zasobów:", A, ",", B)
        return
    payoff_mat = get_matrix_numpy(A, B, fields)
    print("test for for:")
    print("Liczba pól", fields, "liczba zasobów:", A, ",", B)
    epsilons_row = np.zeros((len(phis), len(steps_numbers)))
    epsilons_col = np.zeros((len(phis), len(steps_numbers)))
    times = -np.ones((len(phis), len(steps_numbers)))
    for i in range(len(phis)):
        print(str(int(100*i/len(phis))) + "%")
        for j in range(len(steps_numbers)):
            start_time = time.time()
            strategy_A_row, strategy_B_col = MWU_game_algorithm(A, B, fields, phis[i], steps_numbers[j])
            strategy_B_row, strategy_A_col = MWU_game_algorithm(B, A, fields, phis[i], steps_numbers[j])
            # print(strategy_A_col, strategy_B_col.transpose())
            epsilon_col = epsilon_value(strategy_A_row, strategy_B_row.transpose(), -payoff_mat)
            epsilon_row = epsilon_value(strategy_A_col.transpose(), strategy_B_col, payoff_mat)
            if(epsilon_col == 0 and epsilon_row == 0):
                break
            else:
                epsilons_row[i, j] = epsilon_row
                epsilons_col[i, j] = epsilon_col
                times[i, j] = (time.time() - start_time)/2
        # if(i > 0 and (np.greater_equal(epsilons_col[i,:], epsilons_col[i-1,:])).all()
        #         and (np.greater_equal(epsilons_row[i,:], epsilons_row[i-1,:])).all()):
        #     break
    print("time:", times[times>0].sum())
    times = pd.DataFrame(times, index=phis_names, columns=steps_numbers)
    epsilons_row = pd.DataFrame(epsilons_row, index=phis_names, columns=steps_numbers)
    epsilons_col = pd.DataFrame(epsilons_col, index=phis_names, columns=steps_numbers)
    times.to_csv(RES_PATH_TIMES + name_part + ".csv")
    epsilons_row.to_csv(RES_PATH_ERRORS_ROW + name_part + ".csv")
    epsilons_col.to_csv(RES_PATH_ERRORS_COL + name_part + ".csv")

#%%
fields_MIN = 3
fields_MAX = 11

res_MIN = 10
res_MAX = 26

for res in range(res_MIN,res_MAX):
    for fields in range(fields_MIN,fields_MAX):
        run_epsilon_test(res, res, fields)





