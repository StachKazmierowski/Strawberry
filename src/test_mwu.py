from utils import try_reading_matrix_numpy
from mwu import MWU_game_algorithm
import time
from solutions_evaluator import epsilon_value
from payoff_more_than_half_object import find_and_save_matrix

A = 25
B = 25
n = 9
find_and_save_matrix(A, B, n, None, True)
matrix = -try_reading_matrix_numpy(A,B,n)
print(matrix.shape)
print(matrix)
start = time.time()
results = MWU_game_algorithm(matrix, 1/2, 12801)
print(epsilon_value(results[0], results[1], matrix))

# print("czas: ", time.time() - start)
# print(results[0], results[1])
