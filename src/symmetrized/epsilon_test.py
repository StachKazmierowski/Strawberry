from solutions_evaluator import epsilon_value
from mwu import MWU_game_algorithm
from utils import get_matrix_numpy
import os
import time
import pandas as pd
import numpy as np
RES_PATH_TIMES = "../../res/symmetric/times/"
RES_PATH_ERRORS_COL_COL = "../../res/symmetric/col_col/"
RES_PATH_ERRORS_ROW_ROW = "../../res/symmetric/row_row/"
RES_PATH_ERRORS_COL_ROW = "../../res/symmetric/col_row/"
RES_PATH_ERRORS_ROW_COL = "../../res/symmetric/row_col/"

phis_bound = 11
steps_number_bound = 16
phis = [1/2**i for i in range(1, phis_bound)]
phis_names = ["1/2^" + str(i) for i in range(1, phis_bound)]
steps_numbers = [2**i for i in range(1,steps_number_bound)]

def run_epsilon_test(A=7, B=6, fields=5):
    name_part = str(A) + "_" + str(B) + "_" + str(fields)
    payoff_mat = get_matrix_numpy(A, B, fields)
    print("test for:")
    print("Liczba pól", fields, "liczba zasobów:", A, ",", B)
    epsilons_row_row = np.zeros((len(phis), len(steps_numbers)))
    epsilons_col_col = np.zeros((len(phis), len(steps_numbers)))
    epsilons_row_col = np.zeros((len(phis), len(steps_numbers)))
    epsilons_col_row = np.zeros((len(phis), len(steps_numbers)))
    times = -np.ones((len(phis), len(steps_numbers)))
    if (os.path.exists(RES_PATH_TIMES + name_part + ".csv")):
        times = pd.read_csv(RES_PATH_TIMES + name_part + ".csv", index_col=0)
        epsilons_row_row = pd.read_csv(RES_PATH_ERRORS_ROW_ROW + name_part + ".csv", index_col=0)
        epsilons_col_col = pd.read_csv(RES_PATH_ERRORS_COL_COL + name_part + ".csv", index_col=0)
        epsilons_row_col = pd.read_csv(RES_PATH_ERRORS_ROW_COL + name_part + ".csv", index_col=0)
        epsilons_col_row = pd.read_csv(RES_PATH_ERRORS_COL_ROW + name_part + ".csv", index_col=0)
        times.columns = times.columns.astype(int)
        epsilons_row_row.columns = epsilons_row_row.columns.astype(int)
        epsilons_col_col.columns = epsilons_col_col.columns.astype(int)
        epsilons_row_col.columns = epsilons_row_col.columns.astype(int)
        epsilons_col_row.columns = epsilons_col_row.columns.astype(int)
    else:
        times = pd.DataFrame(times, index=phis_names, columns=steps_numbers)
        epsilons_row_row = pd.DataFrame(epsilons_row_row, index=phis_names, columns=steps_numbers)
        epsilons_col_col = pd.DataFrame(epsilons_col_col, index=phis_names, columns=steps_numbers)
        epsilons_row_col = pd.DataFrame(epsilons_row_col, index=phis_names, columns=steps_numbers)
        epsilons_col_row = pd.DataFrame(epsilons_col_row, index=phis_names, columns=steps_numbers)
    for i in range(len(phis)):
        if (times.iloc[i][2] != -1):
            print("continuing")
            continue
        print(str(int(100*i/len(phis))) + "%")
        for j in range(len(steps_numbers)): #TODO ogarnąć zapisywanie pliku co linia
            start_time = time.time()
            strategy_A_row, strategy_B_col = MWU_game_algorithm(payoff_mat, phis[i], steps_numbers[j])
            strategy_B_row, strategy_A_col = MWU_game_algorithm(-np.transpose(payoff_mat), phis[i], steps_numbers[j])
            # print(strategy_A_col, strategy_B_col.transpose())
            algorith_time = (time.time() - start_time)/2
            epsilon_col = epsilon_value(strategy_A_row, strategy_B_row.transpose(), -payoff_mat)
            epsilon_row = epsilon_value(strategy_A_col.transpose(), strategy_B_col, -payoff_mat)
            epsilon_row_col = epsilon_value(strategy_A_row, strategy_B_col, -payoff_mat)
            epsilon_col_row = epsilon_value(strategy_A_col.transpose(), strategy_B_row.transpose(), -payoff_mat)
            if(epsilon_col == 0 and epsilon_row == 0):
                break
            else:
                epsilons_row_row.iloc[i][2**(j+1)] = epsilon_row
                epsilons_col_col.iloc[i][2**(j+1)] = epsilon_col
                epsilons_row_col.iloc[i][2**(j+1)] = epsilon_row_col
                epsilons_col_row.iloc[i][2**(j+1)] = epsilon_col_row
                times.iloc[i][2**(j+1)] = algorith_time
        times_pd = pd.DataFrame(times, index=phis_names, columns=steps_numbers)
        epsilons_row_row_pd = pd.DataFrame(epsilons_row_row, index=phis_names, columns=steps_numbers)
        epsilons_col_col_pd = pd.DataFrame(epsilons_col_col, index=phis_names, columns=steps_numbers)
        epsilons_row_col_pd = pd.DataFrame(epsilons_row_col, index=phis_names, columns=steps_numbers)
        epsilons_col_row_pd = pd.DataFrame(epsilons_col_row, index=phis_names, columns=steps_numbers)
        times_pd.to_csv(RES_PATH_TIMES + name_part + ".csv")
        epsilons_row_row_pd.to_csv(RES_PATH_ERRORS_ROW_ROW + name_part + ".csv")
        epsilons_col_col_pd.to_csv(RES_PATH_ERRORS_COL_COL + name_part + ".csv")
        epsilons_row_col_pd.to_csv(RES_PATH_ERRORS_ROW_COL + name_part + ".csv")
        epsilons_col_row_pd.to_csv(RES_PATH_ERRORS_COL_ROW + name_part + ".csv")

#%%
fields_MIN = 3
fields_MAX = 11

res_MIN = 10
res_MAX = 26

for res in range(res_MIN,res_MAX):
    for fields in range(fields_MIN,fields_MAX):
        
        run_epsilon_test(res, res, fields)





