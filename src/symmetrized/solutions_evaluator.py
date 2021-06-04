import numpy as np
import pandas as pd
from mwu import MWU_game_algorithm
from utils import try_reading_matrix_numpy, divides, payoff_matrix

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
    res = np.zeros((resource_number+1))
    for i in range(pure_strategies.shape[0]):
        if(strategy[i, 0] > 0):
            pure_strategy = pure_strategies[i,:]
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
