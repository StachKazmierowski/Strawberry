import numpy as np
import pandas as pd
from mwu import MWU_symmetric_game_algorithm
from solutions_parser.chopstic_data_parser import parse_file
from utils import try_reading_symmetric_matrix, try_reading_symmetric_matrix_numpy

path = './data/'
filename = 'tmp.txt'

def eval_symmetric_solution(resource_number, fields_number, phi, steps_number):
    algoritmic_strategy = MWU_symmetric_game_algorithm(resource_number, fields_number, phi, steps_number)
    # row_player_solution = algoritmic_strategy[0]
    column_player_strategy = algoritmic_strategy[1]
    # print(column_player_strategy.shape)
    payoff_mat = try_reading_symmetric_matrix_numpy(resource_number, fields_number)
    row_strategy = chopstic_row_solution_to_vector(6,5,parse_file(path, filename))
    # print(row_strategy)
    # print(np.mu(np.matmul(row_strategy, payoff_mat), (column_player_strategy>0))
    return 0

def chopstic_row_solution_to_vector(resource_number, fields_number, solution):
    matrix_pandas = try_reading_symmetric_matrix(resource_number, fields_number)
    strategy = np.zeros((1,matrix_pandas.shape[0]))
    # print(matrix_pandas.index[3])
    for i in range (matrix_pandas.shape[0]):
        for j in range (solution.shape[0]):
            if(matrix_pandas.index[i] == solution.index[j]):
                strategy[0,i] = solution.iloc[j][0]
    return strategy

def eval_strategy(payoff_matrix, row_solution, column_solution, strategy, game_value=0):
    row_player_strategy = algoritmic_strategy[0]
    column_player_strategy = algoritmic_strategy[1]
    column_solution = column_solution.reshape(column_solution.shape[1], 1)
    row_vector = np.matmul(row_player_strategy, payoff_matrix)
    print(payoff_matrix.shape)
    print(np.multiply(row_vector.reshape(row_vector.shape[1], row_vector.shape[0]), column_solution>0))
    # print(row_solution>0)
    # print(column_solution>0)
    return 0

# Mock data
(resource_number, fields_number, phi, steps_number) = (6,5,1/2,100)
algoritmic_strategy = MWU_symmetric_game_algorithm(resource_number, fields_number, phi, steps_number)
payoff_mat = try_reading_symmetric_matrix_numpy(resource_number, fields_number)
print(payoff_mat.shape)
solution_A = chopstic_row_solution_to_vector(6,5,parse_file(path, filename))
print(eval_strategy(payoff_mat, solution_A, solution_A, algoritmic_strategy))

# chopstic_row_solution_to_vector(6,5,parse_file(path, filename))
# eval_symmetric_solution(6,5,1/2, 1000)

