import time
from src.symmetrized.utils import  payoff_matrix
from src.payoff_matrix_finding.dynamic_payoff import *
from src.payoff_matrix_finding.diff_array import F_0, F_1
import concurrent.futures

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
finder = payoff_dynamic_finder()
# for k in range(2, 5):
#     tmp_mat = finder.payoff_matrix(k, k, k)
#     # print(tmp_mat)
#     tmp_mat_2 = payoff_matrix(k ,k ,k)
#     assert np.all(tmp_mat == tmp_mat_2)
# print("Jest super!")

times = []
for k in range(1, 15):
    start = time.time()
    tmp_mat = finder.payoff_matrix(5 * k, 5 * k, 3)
    times.append(time.time() - start)
    print("czas dla k = ", k, "to", times[-1])
    print("rozmiar macieży to:", tmp_mat.shape)
    print("średni czas obliczania komórki macieży:", times[-1] / (tmp_mat.shape[0]**2))
print(times)
#%%
for i in range(len(times) - 1):
    print(times[i+1] / times[i])
