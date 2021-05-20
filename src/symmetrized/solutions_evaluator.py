import numpy as np
import pandas as pd
from src.symmetrized.mwu import MWU_game_algorithm
from src.solutions_parser.chopstic_data_parser import parse_file
from src.symmetrized.utils import try_reading_matrix_numpy, divides, payoff_matrix

path = './data/'
filename = 'tmp.txt'

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

def find_marginal_distribution(resource_number, fields_number, strategy):
    if(strategy.shape[0] == 1):
        strategy = strategy.reshape((strategy.shape[1], 1))
    pure_strategies = divides(resource_number, fields_number)
    print(pure_strategies.shape)
    print(strategy.shape)
    res = np.zeros((resource_number+1))
    for i in range(pure_strategies.shape[0]):
        if(strategy[i, 0] > 0):
            pure_strategy = pure_strategies[i,:]
            print(pure_strategy)
            print(pure_strategy.shape)
            for j in range(pure_strategies.shape[1]):
                res[int(pure_strategy[j])] += strategy[i,0]/fields_number
    return res

def possible_payoff_increase_B(strategy_A, strategy_B, payoff_mat):
    max_B_payoff = np.matmul(strategy_A, -payoff_mat).max()
    curr_B_payoff = np.matmul(np.matmul(strategy_A, -payoff_mat), strategy_B)[0,0]
    return max_B_payoff - curr_B_payoff

def epsilon_value(strategy_A, strategy_B, payoff_mat):
    epsilon_B = possible_payoff_increase_B(strategy_A, strategy_B, payoff_mat)
    epsilon_A = possible_payoff_increase_B(strategy_B.transpose(), strategy_A.transpose(), -payoff_mat.transpose())
    return max(epsilon_A, epsilon_B)

mock_strategy_A, mock_strategy_B = MWU_game_algorithm(4,4,3,1/16,1000)[1].transpose(), MWU_game_algorithm(4,4,3,1/16,10000)[1]
payoff_mat = payoff_matrix(4,4,3)
# print(possible_payoff_increase_B(mock_strategy_A, mock_strategy_B, payoff_mat))
print(epsilon_value(mock_strategy_A, mock_strategy_B, payoff_mat))
# print(mock_strategy)
# print(find_marginal_distribution(9,3,mock_strategy))
# Mock data
# (resource_number, fields_number, phi, steps_number) = (6,5,1/4,1000)
# algoritmic_strategy = MWU_game_algorithm(resource_number, resource_number, fields_number, phi, steps_number)
# # print(payoff_mat.shape)
# solution_A = chopstic_row_solution_to_vector(6,5,parse_file(path, filename))
# print(solution_A)