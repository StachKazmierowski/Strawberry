from symmetrized.payoff_matrix import *
from symmetrized.diff_array import *
import numpy as np

A = np.array([7, 6, 4, 3, 1, 0])
B = np.array([8, 6, 5, 4, 2, 0])
W, T = single_payoff_matrix_vectors(A,B)

def find_knots(W, T):
    fields_num = W.shape[0]
    i = fields_num
    j = fields_num
    knots = np.array([[fields_num,fields_num]])
    while(i > 0 or j > 0):
        to_print = False
        while(i > 0 and j == W[i-1]):
            i -= 1
            to_print = True
        if(to_print):
            knots = np.append(knots, np.array([[i, j]]).reshape(1,2), axis=0)

        to_print = False
        while(i > 0 and T[i - 1] < j):
            j -= 1
            to_print = True
        if(to_print):
            knots = np.append(knots, np.array([[i, j]]).reshape(1,2), axis=0)

        to_print = False
        while(i > 0 and T[i-1] != W[i-1]):
            i -= 1
            to_print = True
        if(to_print):
            j = W[i]
            knots = np.append(knots, np.array([[i, j]]).reshape(1,2), axis=0)
    return knots

knots = find_knots(W, T)

def m_max(i, j):
    return min(i, j)

def m_min(i, j, fields_num):
    m_2max = min(fields_num - i, j)
    m_4max = min(i, fields_num - j)
    m_234max = m_2max + m_4max + min(fields_num -i-m_2max, fields_num - j - m_4max)
    return fields_num - m_234max

def x_max(i, W):
    return max_rook_num(W[0:i])

def x_min(i, j, W, T):
    return -max_rook_num(L_vector(W[0:i], T[0:i], j))

def find_m_constraints(knots, fields_num):
    m_constraints = np.empty((0, 2))
    for k in range(knots.shape[0]):
        i, j = knots[k][0], knots[k][1]
        m_constraints = np.append(m_constraints, np.array([[m_min(i, j, fields_num), m_max(i, j)]]).reshape(1, 2), axis=0)
    return m_constraints

def find_x_constraints(knots, fields_num):
    x_constraints = np.empty((0, 2))
    for k in range(knots.shape[0]):
        i, j = knots[k][0], knots[k][1]
        x_constraints = np.append(x_constraints, np.array([[x_min(i, j, W, T), x_max(i, W)]]).reshape(1, 2), axis=0)
    return x_constraints

print(find_m_constraints(find_knots(W, T), 6))

print(find_x_constraints(find_knots(W, T), 6))