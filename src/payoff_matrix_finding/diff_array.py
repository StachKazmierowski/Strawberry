from payoff_matrix import *

def F_0(i, j, m, x, flag):
    # print("H0")
    # print(flag)
    if(flag == 1):
        if(x != m):
            return 0
        return single_type_rectangle(i, j, m)
    if(flag == 0):
        if(x != 0):
            return 0
        return single_type_rectangle(i, j, m)
    if(flag == -1):
        if(x != -m):
            return 0
        return single_type_rectangle(i, j, m)

def F_1(i, j, m, x, W, T, flag):
    if(abs(x) > m):
        return 0
    if(m > min(i,j)):
        return 0
    if(flag == -1):
        if(x > 0):
            return 0
        else:
            width = width_to_remove(T)
            return single_type_rectangle(i - width, j, -x) * single_type_rectangle(width, j + x, m + x)
    if(flag == 1):
        if(x < 0):
            return 0
        else:
            height = T[-1] - W[-1]
            return single_type_rectangle(i, j - height, x) * single_type_rectangle(i - x, height, m - x)

def F(i, j, m, x, W, T):
    if(i == 0 or j == 0):
        return 0
    if(m > j or m > i):
        return 0
    if(abs(x) > m):
        return 0
    if(max_rook_num(W) < x):
        return 0
    if(-max_rook_num(L_vector(W, T, j)) > x):
        return 0
    if(is_single_type(W, T, j)):
        return F_0(i, j, m, x, what_single_type(W, T, j))
    if(W[-1] == j): ## corner is in W
        # print("W CORNER")
        width = width_to_remove(W)
        maximum_rooks_in_right = min(width, j)
        sum = 0
        for r in range(min(maximum_rooks_in_right, m) + 1):
            sum += F(i - width, j, m - r, x - r, W[:-width], T[:-width]) * single_type_rectangle(width, j - (m - r), r)
        return sum
    if(T[-1] < j): ## corner is in L
        # print("L CORNER")
        height = j - T[-1]
        maximum_rooks_in_top = min(i, height)
        sum = 0
        for r in range(min(maximum_rooks_in_top, m) + 1):
            F_tmp = F(i, j - height, m - r, x + r, vector_min(W, j - height) , vector_min(T, j - height))
            top = single_type_rectangle(i - (m - r), height, r)
            sum += F_tmp * top
            ## TODO czy nie powinniśmy zmniejszać W i T? done, bez zmian
        return sum
    # REMISY
    if(is_double_type_with_tie(W, T, j)):
        return F_1(i, j, m, x, W, T, what_double_type(W, T, j))
    if(T[-1] == j): ## corner is in T
        # print("WESZLIŚMY W REMISY")
        width = width_to_remove(T)
        height = T[-1] - W[-1]
        maximum_rooks_in_top = min(i - width, height)
        maximum_rooks_in_right = min(width, j - height)
        maximum_rooks_in_corner = min(width, height)
        sum = 0
        for r_3 in range(min(maximum_rooks_in_top, m) + 1):
            for r_2 in range(min(maximum_rooks_in_corner, m) + 1):
                for r_1 in range(min(maximum_rooks_in_right, m) + 1):
                    if(r_1 + r_2 + r_3 <= m):
                        r_4 = m - r_1 - r_2 - r_3
                        F_tmp = F(i - width, j - height, r_4, x - r_1 + r_3, vector_min(W[:-width], j - height)
                                  , vector_min(T[:-width], j - height))
                        top = single_type_rectangle(i - width - r_4, height, r_3)
                        corner = single_type_rectangle(width, height - r_3, r_2)
                        right = single_type_rectangle(width - r_2, j - height - r_4, r_1)
                        sum += F_tmp * top * corner * right
            ## TODO czy nie powinniśmy zmniejszać W i T? done, bez zmian
        return sum

def F_diffs(strategy_one, strategy_two):
    fields_num = strategy_one.shape[0]
    wins = np.zeros((fields_num, 1))
    loses = np.zeros((fields_num, 1))
    W, T = single_payoff_matrix_vectors(strategy_one, strategy_two)
    ties = F(fields_num, fields_num, fields_num, 0, W, T)
    for x in range(fields_num):
        wins[x] = F(fields_num, fields_num, fields_num, x + 1, W, T)
        loses[x] = F(fields_num, fields_num, fields_num, - x - 1, W, T)
    # print(np.sum(wins) + np.sum(loses) + ties)
    # print(factorial(fields_num))
    # assert np.sum(wins) + np.sum(loses) + ties == factorial(fields_num)
    return wins, loses, ties

# x_A = []
# k = 7
# for i in range(k):
#     x_A.append(k - 1 -i)
# print(x_A)
# A = np.array(x_A)
# B = np.array(x_A)
# # print(x_B)
# # #%%
# # A = divides(30, 20)[3]
# # B = divides(30, 20)[3]
# print(A)
# print(B)
# W, T = single_payoff_matrix_vectors(A, B)
# start = time.time()
# H_res = H_diffs(A, B)
# time_H = time.time() - start
# F_res = F_diffs(A, B)
# time_F = time.time() - time_H - start
# # A = symmetrized_pure_payoff_a(A, B)
# # time_pure = time.time() - start - time_F - time_H
# # print(H_res[0] - F_res[0])
# # print(H_res[1] - F_res[1])
# print(F_res)
# print(H_res)
# # print("time P", time_pure)
# print("time H", time_H)
# print("time F", time_F)

# assert(np.sum(H_res[0]) == np.sum(F_res[0]) and np.sum(H_res[1]) == np.sum(F_res[1]) and H_res[2] == F_res[2])

#%%
# print(divides(20, 50).shape)
#%%
# print(0.095801 * 627 * 627 / (3600 * 24))