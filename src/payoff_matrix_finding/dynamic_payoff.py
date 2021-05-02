from src.payoff_matrix_finding.diff_array import *
import numpy as np

def find_knots(W, T):
    fields_num = W.shape[0]
    i = fields_num
    j = fields_num
    knots = np.array([[fields_num,fields_num]])
    while(i > 0 and j > 0):

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
            if(T[i-1] <= W[i]):
                break
        if(to_print):
            j = W[i]
            knots = np.append(knots, np.array([[i, j]]).reshape(1,2), axis=0)
    knots = np.flip(knots, axis=0)
    return knots

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

def find_x_constraints(knots, fields_num): ##TODO do zrobienia
    x_constraints = np.empty((0, 2))
    for k in range(knots.shape[0]):
        i, j = knots[k][0], knots[k][1]
        # x_constraints = np.append(x_constraints, np.array([[x_min(i, j, W, T), x_max(i, W)]]).reshape(1, 2), axis=0)
        x_constraints = np.append(x_constraints, np.array([[-fields_num, fields_num]]).reshape(1, 2), axis=0)
    return x_constraints