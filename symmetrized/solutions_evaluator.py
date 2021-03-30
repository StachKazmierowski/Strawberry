import numpy as np
import pandas as pd
from symmetrized.mwu import MWU_symmetric_game_algorithm
from solutions_parser.chopstic_data_parser import parse_file
from symmetrized.utils import try_reading_symmetric_matrix, try_reading_symmetric_matrix_numpy, divides

path = './data/'
filename = 'tmp.txt'

def eval_symmetric_solution(resource_number, fields_number, phi, steps_number):
    algoritmic_strategy = MWU_symmetric_game_algorithm(resource_number, fields_number, phi, steps_number)
    row_player_solution = algoritmic_strategy[0]
    column_player_strategy = algoritmic_strategy[1]
    payoff_mat = try_reading_symmetric_matrix_numpy(resource_number, fields_number)
    row_strategy = chopstic_row_solution_to_vector(6,5,parse_file(path, filename))
    return 0

def chopstic_row_solution_to_vector(resource_number, fields_number, solution):
    pure_strategies = divides(resource_number, fields_number)
    strategy = np.zeros((1,pure_strategies.shape[0]))
    for i in range (pure_strategies.shape[0]):
        for j in range (solution.shape[0]):
            if(str(pure_strategies[i]) == solution.index[j]):
                strategy[0,i] = solution.iloc[j][0]
    return strategy

def eval_strategy(payoff_matrix, row_solution, column_solution, algoritmic_strategy, game_value=0):
    row_player_strategy = algoritmic_strategy[0]
    column_player_strategy = algoritmic_strategy[1]
    column_solution = column_solution.reshape(column_solution.shape[1], 1)
    row_vector = np.matmul(row_player_strategy, payoff_matrix)
    column_vector = np.matmul(payoff_matrix, column_player_strategy)
    column_biggest_error = np.max((abs(game_value - np.multiply(row_vector.reshape(row_vector.shape[1], row_vector.shape[0]), column_solution>0)))[column_solution>0])
    row_biggest_error = np.max(abs(game_value - np.multiply(row_solution > 0, column_vector.reshape(column_vector.shape[1], column_vector.shape[0])))[row_solution > 0])
    return column_biggest_error, row_biggest_error

# Mock data
# (resource_number, fields_number, phi, steps_number) = (6,5,1/4,1000)
# algoritmic_strategy = MWU_symmetric_game_algorithm(resource_number, fields_number, phi, steps_number)
# payoff_mat = try_reading_symmetric_matrix_numpy(resource_number, fields_number)
# # print(payoff_mat.shape)
# solution_A = chopstic_row_solution_to_vector(6,5,parse_file(path, filename))
# print(solution_A)
