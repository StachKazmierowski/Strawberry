import time
from dynamic_payoff import find_m_constraints, find_knots, find_x_constraints
from payoff_matrix import single_payoff_matrix_vectors, vector_min, max_rook_num, L_vector, \
    is_single_type, what_single_type, single_type_rectangle, width_to_remove, is_double_type_with_tie, what_double_type
from diff_array import F_0, F_1
import concurrent.futures
from utils import divides
import numpy as np
import pandas as pd

class payoff_dynamic_finder:
    def __init__(self):
        return

    def reload(self, A, B):
        self.A = A
        self.B = B
        self.fields_number = A.shape[0]
        self.W, self.T = single_payoff_matrix_vectors(A, B)
        self.knots = find_knots(self.W, self.T)
        self.m_constraints = find_m_constraints(self.knots, self.fields_number).astype(int)
        self.x_constraints = find_x_constraints(self.knots, self.fields_number).astype(int)
        # self.x_contraints = np  TODO zrobić ograniczenia na x
        self.x_min = np.min(self.x_constraints)
        self.x_max = np.max(self.x_constraints)
        self.x_range = int(np.max(self.x_constraints) - np.min(self.x_constraints))
        ## i, j, m, x order
        self.values = np.zeros((self.fields_number + 1, self.fields_number + 1, self.fields_number + 1, self.x_range + 1))

    def run_experiment(self):
        assert (self.knots.shape) == (self.m_constraints.shape) == (self.x_constraints.shape)
        for knot_index in range(self.knots.shape[0]):
            for m in range(self.m_constraints[knot_index, 0], self.m_constraints[knot_index, 1] + 1):
                for x in range(self.x_constraints[knot_index, 0], self.x_constraints[knot_index, 1] + 1):
                    i = self.knots[knot_index, 0]
                    j = self.knots[knot_index, 1]
                    self.values[i, j, m, x] = self.F(i, j, m, x)

    def F(self, i, j, m, x):
        if(i == 0 or j == 0):
            return 0
        if(m > j or m > i):
            return 0
        if(abs(x) > m):
            return 0
        W_tmp = vector_min(self.W, j)[:i]
        T_tmp = vector_min(self.T, j)[:i]
        if(max_rook_num(W_tmp) < x):
            return 0
        if(-max_rook_num(L_vector(W_tmp, T_tmp, j)) > x):
            return 0
        if(is_single_type(W_tmp, T_tmp, j)):
            return F_0(i, j, m, x, what_single_type(W_tmp, T_tmp, j))
        if(W_tmp[-1] == j): ## corner is in W
            width = width_to_remove(W_tmp)
            maximum_rooks_in_right = min(width, j)
            sum = 0
            for r in range(min(maximum_rooks_in_right, m) + 1):
                sum += self.values[i - width, j, m - r, x - r] * single_type_rectangle(width, j - (m - r), r)
            return sum
        if(T_tmp[-1] < j): ## corner is in L
            height = j - T_tmp[-1]
            maximum_rooks_in_top = min(i, height)
            sum = 0
            for r in range(min(maximum_rooks_in_top, m) + 1):
                F_tmp = self.values[i, j - height, m - r, x + r]
                top = single_type_rectangle(i - (m - r), height, r)
                sum += F_tmp * top
            return sum
        # REMISY
        if(is_double_type_with_tie(W_tmp, T_tmp, j)):
            return F_1(i, j, m, x, W_tmp, T_tmp, what_double_type(W_tmp, T_tmp, j))
        if(T_tmp[-1] == j): ## corner is in T
            width = width_to_remove(T_tmp)
            height = T_tmp[-1] - W_tmp[-1]
            maximum_rooks_in_top = min(i - width, height)
            maximum_rooks_in_right = min(width, j - height)
            maximum_rooks_in_corner = min(width, height)
            sum = 0
            for r_3 in range(min(maximum_rooks_in_top, m) + 1):
                for r_2 in range(min(maximum_rooks_in_corner, m) + 1):
                    for r_1 in range(min(maximum_rooks_in_right, m) + 1):
                        if(r_1 + r_2 + r_3 <= m):
                            r_4 = m - r_1 - r_2 - r_3
                            F_tmp = self.values[i - width, j - height, r_4, x - r_1 + r_3]
                            top = single_type_rectangle(i - width - r_4, height, r_3)
                            corner = single_type_rectangle(width, height - r_3, r_2)
                            right = single_type_rectangle(width - r_2, j - height - r_4, r_1)
                            sum += F_tmp * top * corner * right
            return sum

    def payoff(self, iter):
        i = iter[0]
        j = iter[1]
        A = self.A_symmetrized_strategies[i]
        B = self.B_symmetrized_strategies[j]
        self.reload(A,B)
        self.run_experiment()
        wins = self.values[-1, -1, -1, self.x_constraints[-1, 0] :].sum()
        loses = self.values[-1, -1, -1, 1 : self.x_constraints[-1, 1] + 1].sum()
        ties = self.values[-1, -1, -1, 0]
        if(wins+loses+ties == 0):
            print(wins, loses, ties)
        return i, j, (wins - loses) / ( wins + loses + ties)

    def payoff_matrix(self, A_number, B_number, n): ## TODO dla symetrycznej gry wypełniamy tylko pół macieży
        symmetric = (A_number == B_number)
        self.A_symmetrized_strategies = divides(A_number, n)
        self.B_symmetrized_strategies = divides(B_number, n)
        matrix = np.zeros((self.A_symmetrized_strategies.shape[0], self.B_symmetrized_strategies.shape[0]))
        if(symmetric):
            args = ((i,j) for i in range(self.A_symmetrized_strategies.shape[0]) for j in range(i))
            with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
                for i, j, val in executor.map(self.payoff, args):
                    matrix[i, j] = val
                    matrix[j, i] = -val
        else:
            args = ((i,j) for i in range(self.A_symmetrized_strategies.shape[0]) for j in range(self.B_symmetrized_strategies.shape[0]))
            with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
                for i, j, val in executor.map(self.payoff, args):
                    matrix[i, j] = val
        return matrix


#%%
def pd_payoff_matrix(matrix, A, B, n):
    A_symmetrized_strategies = divides(A, n)
    B_symmetrized_strategies = divides(B, n)
    columns_names = []
    rows_names = []
    for i in range(A_symmetrized_strategies.shape[0]):
        rows_names.append(str(A_symmetrized_strategies[i]))
    for i in range(B_symmetrized_strategies.shape[0]):
        columns_names.append(str(B_symmetrized_strategies[i]))
    df = pd.DataFrame(matrix, columns=columns_names, index=rows_names)
    return df

def save_matrix_pd(A, B, n, df):
    df.to_csv("../../data/payoff_matrices_dynamic/payoff_matrix(" + str(A) + "," + str(B) + "," + str(n) + ").csv")
    if(A != B):
        (-df.transpose()).to_csv("./data/payoff_matrices_dynamic/payoff_matrix(" + str(B) + "," + str(A) + "," + str(n) + ").csv")


finder = payoff_dynamic_finder()
#%%
K_MIN = 1
K_MAX = 5

fields_MIN = 3
fields_MAX = 11

res_MIN = 10
res_MAX = 26
def save_times_pandas(times, name):
    columns_names = []
    rows_names = []
    for i in range(res_MIN, res_MAX):
        rows_names.append(i)
    for i in range(fields_MIN, fields_MAX):
        columns_names.append(i)
    df = pd.DataFrame(times, columns=columns_names, index=rows_names)
    df.to_csv("../../data/times/" + name + ".csv")

def save_times(times, name):
    times.to_csv("../../data/times/" + name + ".csv")

save_times_pandas(np.zeros((16,8)), "time_iter")
save_times_pandas(np.zeros((16,8)), "cell_time_iter")
#%%
if __name__ == '__main__':
    # times = pd.read_csv("../../data/times/time.csv", index_col=0)
    # times_per_cell = pd.read_csv("../../data/times/cell_time.csv", index_col=0)
    # for k in range(K_MIN, K_MAX):
    #     for fields in range(fields_MIN, fields_MAX):
    #         if(times.iloc[k-K_MIN][fields-fields_MIN] != 0):
    #             print("already have value for:")
    #             print("Liczba pól", fields, "liczba zasobów:", fields * k, "czas", times.iloc[k-K_MIN][fields-fields_MIN])
    #             if(times.iloc[k-K_MIN][fields-fields_MIN] > 10 * 60 ):
    #                 break
    #             else:
    #                 continue
    #         start = time.time()
    #         np_mat = finder.payoff_matrix(k * fields, k * fields, fields)
    #         delta_time = time.time() - start
    #         times.iloc[k-K_MIN][fields-fields_MIN]= delta_time
    #         times_per_cell.iloc[k-K_MIN][fields-fields_MIN] = (delta_time / (np_mat.shape[0]**2))
    #         pd_mat = pd_payoff_matrix(np_mat, k * fields, k * fields, fields)
    #         save_matrix_pd(k * fields, k * fields, fields, pd_mat)
    #         save_times(times, "time")
    #         save_times(times_per_cell, "cell_time")
    #         print("new value for:")
    #         print("Liczba pól", fields, "liczba zasobów:", fields * k, "czas", delta_time)
    #         if(delta_time > 10 * 60):
    #             break
    # print(times)
    # print(times_per_cell)
    times = pd.read_csv("../../data/times/time_iter.csv", index_col=0)
    times_per_cell = pd.read_csv("../../data/times/cell_time_iter.csv", index_col=0)
    for fields in range(fields_MIN, fields_MAX):
        for res in range(res_MIN, res_MAX):
            if(times.iloc[res-res_MIN][fields-fields_MIN] != 0):
                print("already have value for:")
                print("Liczba pól", fields, "liczba zasobów:", res, "czas", times.iloc[res-res_MIN][fields-fields_MIN])
                if(times.iloc[res-res_MIN][fields-fields_MIN] > 10 * 60 ):
                    break
                else:
                    continue
            start = time.time()
            np_mat = finder.payoff_matrix(res, res, fields)
            delta_time = time.time() - start
            times.iloc[res-res_MIN][fields-fields_MIN]= delta_time
            times_per_cell.iloc[res-res_MIN][fields-fields_MIN] = (delta_time / (np_mat.shape[0]**2))
            pd_mat = pd_payoff_matrix(np_mat, res, res, fields)
            save_matrix_pd(res, res, fields, pd_mat)
            save_times(times, "time")
            save_times(times_per_cell, "cell_time")
            print("new value for:")
            print("Liczba pól", fields, "liczba zasobów:", res, "czas", delta_time)
            if(delta_time > 10 * 60):
                break
