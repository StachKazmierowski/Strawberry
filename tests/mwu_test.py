from symmetrized.solutions_evaluator import eval_strategy, chopstic_row_solution_to_vector
from symmetrized.mwu import MWU_symmetric_game_algorithm, MWU_game_algorithm
from symmetrized.utils import try_reading_symmetric_matrix_numpy, try_reading_matrix_numpy
from solutions_parser.chopstic_data_parser import parse_file, parse_game_value
import time
import pandas as pd
import numpy as np
RES_PATH_TIMES = "./results/symmetric/times/"
RES_PATH_ERRORS_COL = "./results/symmetric/errors_column/"
RES_PATH_ERRORS_ROW = "./results/symmetric/errors_row/"
SOLUTION_PATH = "./data/results4/"

phis_bound = 2
steps_number_bound = 5
phis = [1/2**i for i in range(1, phis_bound)]
phis_names = ["1/2^" + str(i) for i in range(1, phis_bound)]
steps_numbers = [10**i for i in range(1,steps_number_bound)]

def run_test_symmetric(resources=6, fields=5):
    payoff_matrix = try_reading_symmetric_matrix_numpy(resources, fields)
    name_part = str(fields) + "_" + str(resources) + "_" + str(resources)
    solution_path = SOLUTION_PATH + "battlefiels_" + str(fields) + "/results_" + str(name_part) + "/"
    solution_file = "optimal_strategy_" + str(name_part) + "_A.txt"
    solution_A = chopstic_row_solution_to_vector(resources, fields, parse_file(solution_path, solution_file))
    solution_file = "optimal_strategy_" + str(name_part) + "_B.txt"
    solution_B = chopstic_row_solution_to_vector(resources, fields, parse_file(solution_path, solution_file))
    errors_row = np.zeros((len(phis), len(steps_numbers)))
    errors_col = np.zeros((len(phis), len(steps_numbers)))
    times = -np.ones((len(phis), len(steps_numbers)))
    for i in range(len(phis)):
        for j in range(len(steps_numbers)):
            start_time = time.time()
            error = eval_strategy(payoff_matrix, solution_A, solution_B,
                                         MWU_symmetric_game_algorithm(resources, fields, phis[i], steps_numbers[j]))
            if(error[0] == 0 and error[1] == 0):
                break
            else:
                errors_col[i, j] = error[0]
                errors_row[i, j] = error[1]
                times[i, j] = time.time() - start_time
        if(i > 0 and (np.greater_equal(errors_row[i,:], errors_row[i-1,:])).all()
                and (np.greater_equal(errors_col[i,:], errors_col[i-1,:])).all()):
            # print(i)
            break
    times = pd.DataFrame(times, index=phis_names, columns=steps_numbers)
    errors_row = pd.DataFrame(errors_row, index=phis_names, columns=steps_numbers)
    errors_col = pd.DataFrame(errors_col, index=phis_names, columns=steps_numbers)
    times.to_csv(RES_PATH_TIMES + name_part + ".csv")
    errors_row.to_csv(RES_PATH_ERRORS_ROW + name_part + ".csv")
    errors_col.to_csv(RES_PATH_ERRORS_COL + name_part + ".csv")

def run_test(A=7, B=6, fields=5):
    payoff_matrix = try_reading_matrix_numpy(A, B, fields)
    name_part = str(fields) + "_" + str(A) + "_" + str(B)
    solution_path = SOLUTION_PATH + "battlefiels_" + str(fields) + "/results_" + str(name_part) + "/"
    solution_file = "optimal_strategy_" + str(name_part) + "_A.txt"
    solution_A = chopstic_row_solution_to_vector(A, fields, parse_file(solution_path, solution_file))
    solution_file = "optimal_strategy_" + str(name_part) + "_B.txt"
    solution_B = chopstic_row_solution_to_vector(B, fields, parse_file(solution_path, solution_file))
    game_value_file = "expected_payoff_" + str(name_part) + "_B.txt"
    game_value = parse_game_value(solution_path, game_value_file)
    print(game_value)
    errors_row = np.zeros((len(phis), len(steps_numbers)))
    errors_col = np.zeros((len(phis), len(steps_numbers)))
    times = -np.ones((len(phis), len(steps_numbers)))
    for i in range(len(phis)):
        for j in range(len(steps_numbers)):
            start_time = time.time()
            error = eval_strategy(payoff_matrix, solution_A, solution_B,
                                         MWU_game_algorithm(A, B, fields, phis[i], steps_numbers[j]), game_value)
            if(error[0] == 0 and error[1] == 0):
                break
            else:
                errors_col[i, j] = error[0]
                errors_row[i, j] = error[1]
                times[i, j] = time.time() - start_time
        if(i > 0 and (np.greater_equal(errors_row[i,:], errors_row[i-1,:])).all()
                and (np.greater_equal(errors_col[i,:], errors_col[i-1,:])).all()):
            # print(i)
            break
    print(errors_col)
    print(errors_row)
    # print(errors_row)
    # times = pd.DataFrame(times, index=phis_names, columns=steps_numbers)
    # errors_row = pd.DataFrame(errors_row, index=phis_names, columns=steps_numbers)
    # errors_col = pd.DataFrame(errors_col, index=phis_names, columns=steps_numbers)
    # times.to_csv(RES_PATH_TIMES + name_part + ".csv")
    # errors_row.to_csv(RES_PATH_ERRORS_ROW + name_part + ".csv")
    # errors_col.to_csv(RES_PATH_ERRORS_COL + name_part + ".csv")

# for res in range(2,11):
#     for fields in range(2,11):
#         if(res > fields):
#             start = time.time()
#             run_test(res, fields)
#             print("Liczba zasobów:", res, "Liczba pól:", fields)
#             print("Dominika jest hooot, żeby to obliczyć potrzebne było: ", time.time()-start, "sekund")

run_test(7,6,5)




