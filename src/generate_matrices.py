import numpy as np
import time
import pandas as pd
from os import path
from payoff_more_than_half_object import payoff_dynamic_finder_more_than_half, payoff_matrix_pd, find_and_save_matrix

A_MIN = 10
A_MAX = 12
n_min = 3
n_max = 5
n_s = [n for n in range(n_min, n_max + 1, 2)]
A_s = [A for A in range(A_MIN, A_MAX + 1)]

if(path.exists("../matrix_creation_times_dev/times_n=(" + str(n_min) + "," + str(n_max) + ")" +"_A=(" + str(A_MIN) + "," + str(A_MAX) + ").csv")):
    times = pd.read_csv("../matrix_creation_times_dev/times_n=(" + str(n_min) + "," + str(n_max) + ")" +"_A=(" + str(A_MIN) + "," + str(A_MAX) + ").csv", index_col=0)
    times.columns = A_s
    times.index = n_s
    print(times)
else:
    times = np.zeros((len(n_s), len(A_s)))
    times = pd.DataFrame(times, index=n_s, columns=A_s)

for n in range(n_min, n_max + 1, 2):
    for A in range(A_MIN, A_MAX + 1):
        print("n = ", n, ", A = ", A)
        start = time.time()
        find_and_save_matrix(A, A, n, None, True)
        end = time.time()
        if(end - start > 0.1):
            times[A][n] = (end - start)
            print("time = ", end - start)
            times.to_csv("../matrix_creation_times_dev/times_n=(" + str(n_min) + "," + str(n_max) + ")" +"_A=(" + str(A_MIN) + "," + str(A_MAX) + ").csv")
print(times)