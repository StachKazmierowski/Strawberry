import pandas as pd
import numpy as np
import time
import sys
from payoff_object import payoff_dynamic_finder, pd_payoff_matrix

fields_MIN, fields_MAX = 5, 8
res_MIN, res_MAX = 5, 15

def save_times(times, name):
    times.to_csv("../../data/threads_times/" + name + ".csv")

if __name__ == '__main__':
    print(sys.argv[1])
    fields_MIN, fields_MAX = int(sys.argv[1]), int(sys.argv[2])
    print(fields_MIN, fields_MAX)
    finder = payoff_dynamic_finder()
    times = np.zeros((4, fields_MAX-fields_MIN))
    times_per_cell = np.zeros((4, fields_MAX-fields_MIN))
    for thread in range(4):
        threads_num = 2**thread
        for fields in range(fields_MIN, fields_MAX):
            res = 10
            start = time.time()
            np_mat = finder.payoff_matrix(res, res, fields,threads_num)
            delta_time = time.time() - start
            times[thread, fields-fields_MIN] = delta_time
            times_per_cell[thread, fields-fields_MIN] = (delta_time / (np_mat.shape[0]**2))
            print(threads_num, fields, delta_time)
    rows = [2**i for i in range(4)]
    columns = [i for i in range(fields_MIN, fields_MAX)]
    print(rows)
    print(columns)
    pd_times = pd.DataFrame(times, index=rows, columns=columns)
    pd_times_per_cell = pd.DataFrame(times_per_cell, index=rows, columns=columns)
    save_times(pd_times, "time_threads_fields")
    save_times(pd_times_per_cell, "time_threads_fields_per_cell")
        # for res in range(res_MIN, res_MAX):
        #     fields = 10
        #     if(times.iloc[res-res_MIN][fields-fields_MIN] != 0):
        #         print("already have value for:")
        #         print("Liczba pól", fields, "liczba zasobów:", res, "czas", times.iloc[res-res_MIN][fields-fields_MIN])
        #         if(times.iloc[res-res_MIN][fields-fields_MIN] > 10 * 60 ):
        #             break
        #         else:
        #             continue
        #     start = time.time()
        #     np_mat = finder.payoff_matrix(res, res, fields,)
        #     delta_time = time.time() - start
        #     times.iloc[res-res_MIN][fields-fields_MIN]= delta_time
        #     times_per_cell.iloc[res-res_MIN][fields-fields_MIN] = (delta_time / (np_mat.shape[0]**2))
        #     pd_mat = pd_payoff_matrix(np_mat, res, res, fields)
        #     # save_matrix_pd(res, res, fields, pd_mat)
        #     save_times(times, "time_iter")
        #     save_times(times_per_cell, "cell_time_iter")
        #     print("new value for:")
        #     print("Liczba pól", fields, "liczba zasobów:", res, "czas", delta_time)
        #     if(delta_time > 10 * 60):
        #         break