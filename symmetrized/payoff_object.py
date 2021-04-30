import numpy as np
from symmetrized.payoff_matrix import *
from symmetrized.payoff_matrix_tests import H_diffs
import time
from symmetrized.utils import symmetrized_pure_payoff_a, payoff_matrix
from symmetrized.dynamic_payoff import *
from symmetrized.diff_array import F_0, F_1

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
        # self.x_contraints = np
        self.x_min = np.min(self.x_constraints)
        self.x_max = np.max(self.x_constraints)
        self.x_range = int(np.max(self.x_constraints) - np.min(self.x_constraints) + 1)
        ## i, j, m, x order
        self.values = np.zeros((self.fields_number + 1, self.fields_number + 1, self.fields_number + 1, self.x_range + 1))


    def run_experiment(self):
        assert (self.knots.shape) == (self.m_constraints.shape) == (self.x_constraints.shape)
        print(self.A, self.B)
        print(self.W)
        print(self.T)
        print(self.knots)
        # print("x_constraints", self.x_constraints)
        for knot_index in range(self.knots.shape[0]):
            for m in range(self.m_constraints[knot_index, 0], self.m_constraints[knot_index, 1] + 1):
                for x in range(self.x_constraints[knot_index, 0], self.x_constraints[knot_index, 1] + 1):
                    i = self.knots[knot_index, 0]
                    j = self.knots[knot_index, 1]
                    print("seeking values for:", i, j, m, x)
                    self.values[i, j, m, x] = self.F(i, j, m, x)
                    print(self.values[i, j, m, x])
        # print(self.values[-1,-1,:,:])

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
                ## TODO czy nie powinniśmy zmniejszać W i T? done, bez zmian
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
                ## TODO czy nie powinniśmy zmniejszać W i T? done, bez zmian
            return sum

    def F_diffs(self):
        # print(self.fields_number)
        # print(A)
        fields_num = self.A.shape[0]
        wins = np.zeros((fields_num, 1))
        loses = np.zeros((fields_num, 1))
        ties = self.values[fields_num, fields_num, fields_num, 0]
        for x in range(self.x_constraints[-1, 0], self.x_constraints[-1, 1] + 1):
            if(x < 0):
                loses[x] = self.values[fields_num, fields_num, fields_num, x]
            elif(x > 0):
                wins[x - 1] = self.values[fields_num, fields_num, fields_num, x]
        # print(self.values[-1, -1, -1, :])
        return wins, loses, ties

    def payoff(self):
        wins, loses, ties = self.F_diffs()
        # print("results:", wins.sum(), loses.sum(), ties)
        return - (wins.sum() - loses.sum()) / (wins.sum() + loses.sum() + ties)

    def payoff_matrix(self, A_number, B_number, n):
        A_symmetrized_strategies = divides(A_number, n)
        B_symmetrized_strategies = divides(B_number, n)
        matrix = np.zeros((A_symmetrized_strategies.shape[0], B_symmetrized_strategies.shape[0]))
        for i in range(A_symmetrized_strategies.shape[0]):
            for j in range(B_symmetrized_strategies.shape[0]):
                if(np.all(A_symmetrized_strategies[i] == B_symmetrized_strategies[j])):
                    matrix[i, j] = 0
                else:
                    self.reload(A_symmetrized_strategies[i], B_symmetrized_strategies[j])
                    self.run_experiment()
                    if(i == 1 and j ==2):
                        print("wektory:", self.A, self.B)
                    matrix[i, j] = self.payoff()
                    if(self.payoff() != symmetrized_pure_payoff_a(self.A, self.B)):
                        print("error for", self.A, self.B)
        return matrix

# x_A = []
# k = 6
# for i in range(k):
#     x_A.append(k - 1 -i)
# print(x_A)
# A = np.array(x_A)
# B = np.array(x_A)
# A = np.array([7, 6, 4, 3, 1, 0])
# B = np.array([8, 6, 5, 4, 2, 0])
finder = payoff_dynamic_finder()
A = np.array([3,1,0,0])
B = np.array([2,1,1,0])
finder.reload(A, B)
finder.run_experiment()
print(finder.F_diffs())
# print("time:", time.time() - start)
print(finder.payoff())
print(symmetrized_pure_payoff_a(A, B))

#%%
finder = payoff_dynamic_finder()
k = 5
tmp_mat = finder.payoff_matrix(k, k, k)
print("diff at positiuons: ", (tmp_mat != payoff_matrix(k, k, k)).sum())
print(tmp_mat != payoff_matrix(k, k, k))
print(tmp_mat)


