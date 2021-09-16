import time
import datetime
from dynamic_payoff import find_m_constraints, find_knots
from payoff_matrix import single_payoff_matrix_vectors, vector_min, max_rook_num, L_vector, \
    is_single_type, what_single_type, single_type_rectangle, width_to_remove, is_double_type_with_tie, what_double_type, \
    single_payoff_matrix
from diff_array import how_many_rooks_permutations_in_W_single_area, how_many_rooks_permutations_in_W_double_area
import concurrent.futures
from utils import divides
import numpy as np
import pandas as pd
import os
from os import path
from utils import MATRICES_PATH, MATRICES_RAPORTS_PATH

class payoff_dynamic_finder_more_than_half():
    def __init__(self):
        return

    def reload(self, A, B):
        self.A = A
        self.B = B
        self.fields_number = A.shape[0]
        if(int(self.fields_number / 2) * 2 == self.fields_number):
            print("Even number of fields!!!")
            return
        self.W, self.T = single_payoff_matrix_vectors(A, B)
        self.knots = find_knots(self.W, self.T)
        self.m_constraints = find_m_constraints(self.knots, self.fields_number).astype(int)
        self.x_max = max_rook_num(self.W)
        ## i, j, m, x order
        self.values = np.zeros((self.fields_number + 1, self.fields_number + 1, self.fields_number + 1, self.x_max + 1))

    def run_experiment(self):
        assert (self.knots.shape) == (self.m_constraints.shape)
        for knot_index in range(self.knots.shape[0]):
            for m in range(self.m_constraints[knot_index, 0], self.m_constraints[knot_index, 1] + 1):
                for x in range(self.x_max + 1):
                    i = self.knots[knot_index, 0]
                    j = self.knots[knot_index, 1]
                    self.values[i, j, m, x] = self.F(i, j, m, x)

    def F(self, i, j, m, x):
        if(i == 0 or j == 0):
            return 0
        if(m > j or m > i):
            return 0
        if(x > m or x < 0):
            return 0
        W_tmp = vector_min(self.W, j)[:i]
        T_tmp = vector_min(self.T, j)[:i]
        if(max_rook_num(W_tmp) < x):
            return 0
        if(is_single_type(W_tmp, T_tmp, j)):
            return how_many_rooks_permutations_in_W_single_area(i, j, m, x, what_single_type(W_tmp, T_tmp, j))
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
                F_tmp = self.values[i, j - height, m - r, x]
                top = single_type_rectangle(i - (m - r), height, r)
                sum += F_tmp * top
            return sum
        # REMISY
        if(is_double_type_with_tie(W_tmp, T_tmp, j)):
            return how_many_rooks_permutations_in_W_double_area(i, j, m, x, W_tmp, T_tmp, what_double_type(W_tmp, T_tmp, j))
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
                            F_tmp = self.values[i - width, j - height, r_4, x - r_1]
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
        wins = self.values[-1,-1,-1, int( (self.fields_number + 1) / 2):].sum()

        self.rotate_clash_matrix()
        self.run_experiment()
        loses = self.values[-1,-1,-1, int( (self.fields_number + 1) / 2):].sum()

        number_of_permutations = single_type_rectangle(self.fields_number, self.fields_number, self.fields_number)
        return i, j, (wins - loses) / number_of_permutations

    def single_payoff(self, A, B):
        self.reload(A,B)
        self.run_experiment()
        wins = self.values[-1,-1,-1, int( (self.fields_number + 1) / 2):].sum()

        self.rotate_clash_matrix()
        self.run_experiment()
        loses = self.values[-1,-1,-1, int( (self.fields_number + 1) / 2):].sum()

        number_of_permutations = single_type_rectangle(self.fields_number, self.fields_number, self.fields_number)
        return (wins - loses) / number_of_permutations

    def rotate_clash_matrix(self):
        tmp_W = self.fields_number - np.flip(self.T, 0) ## to jest nowe W
        self.T = self.fields_number - np.flip(self.W, 0) ## to jest nowe T
        self.W = tmp_W
        self.knots = find_knots(self.W, self.T)
        self.m_constraints = find_m_constraints(self.knots, self.fields_number).astype(int)
        self.x_max = np.max(max_rook_num(self.W))
        ## i, j, m, x order
        self.values = np.zeros((self.fields_number + 1, self.fields_number + 1, self.fields_number + 1, self.x_max + 1))

    def payoff_matrix(self, A_number, B_number, n, threads_number=None):
        symmetric = (A_number == B_number)
        self.A_symmetrized_strategies = divides(A_number, n)
        self.B_symmetrized_strategies = divides(B_number, n)
        matrix = np.zeros((self.A_symmetrized_strategies.shape[0], self.B_symmetrized_strategies.shape[0]))
        if(symmetric):
            args = ((i,j) for i in range(self.A_symmetrized_strategies.shape[0]) for j in range(i)) ## Nie robimy przekątnej bo zawsze są na niej zera
            with concurrent.futures.ProcessPoolExecutor(max_workers=threads_number) as executor:
                for i, j, val in executor.map(self.payoff, args):
                    matrix[i, j] = val
                    matrix[j, i] = -val
        else:
            args = ((i,j) for i in range(self.A_symmetrized_strategies.shape[0]) for j in range(self.B_symmetrized_strategies.shape[0]))
            with concurrent.futures.ProcessPoolExecutor(max_workers=threads_number) as executor:
                for i, j, val in executor.map(self.payoff, args):
                    matrix[i, j] = val
        return -matrix


def payoff_matrix_pd(matrix, A, B, n):
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
    dir_path = MATRICES_PATH + str(n) + "_fields"
    raport_path = MATRICES_RAPORTS_PATH + str(n) + "_fields"
    if(not path.exists(dir_path)):
        os.makedirs(dir_path)
        os.makedirs(raport_path)
    df.to_csv(MATRICES_PATH + str(n) + "_fields" + "/payoff_matrix(" + str(A) + "," + str(B) + "," + str(n) + ").csv")
    if(A != B):
        (-df.transpose()).to_csv(MATRICES_PATH + str(n) + "_fields" + "/payoff_matrix(" + str(B) + "," + str(A) + "," + str(n) + ").csv")

def check_if_matrix_exists(A,B,n):
    dir_path = MATRICES_PATH + str(n) + "_fields"  + "/payoff_matrix(" + str(A) + "," + str(B) + "," + str(n) + ").csv"
    if(not path.exists(dir_path)):
        return False
    return True

def find_and_save_matrix(A,B,n, threads_count=None):
    if(check_if_matrix_exists(A,B,n)):
       return
    finder = payoff_dynamic_finder()
    start_time = time.time()
    payoff_mat_np = finder.payoff_matrix(A,B,n, threads_count)
    delta_time = time.time() - start_time
    payoff_mat_pd = payoff_matrix_pd(payoff_mat_np, A,B,n)
    save_matrix_pd(A,B,n,payoff_mat_pd)
    save_raport(A,B,n,delta_time,threads_count)
    save_raport(B,A,n,delta_time,threads_count)
    return

def save_raport(A,B,n,delta_time,threads_count):
    f = open(MATRICES_RAPORTS_PATH + str(n) + "_fields/" + "payoff_matrix_raport(" + str(B) + "," + str(A) + "," + str(n) + ").txt", "w")
    now = datetime.datetime.now()
    f.write("Macierz obliczono: " + str(now) + "\n")
    f.write("Czas obliczania: " + str(delta_time) + "s\n")
    f.write("Liczba użytych wątków: ")
    if(threads_count==None):
        f.write("domyślna (nie podano)")
    else:
        f.write(str(threads_count))
    f.close()

