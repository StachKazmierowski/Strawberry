import numpy as np
import pandas as pd
from itertools import permutations

def next_divide(divide):
    n = divide.shape[1]
    div_num = divide.shape[0]
    dev_tmp = np.empty((0, n), int)
    for i in range(div_num):
        tmp = divide[i][:]
        for j in range(n):
            if (j == 0 or tmp[j] < tmp[j - 1]):
                tmp[j] = tmp[j] + 1
                dev_tmp = np.append(dev_tmp, tmp.reshape(1, n), axis=0)
                tmp[j] = tmp[j] - 1
    return (np.unique(dev_tmp, axis=0))

def divides(A, n):
    if (A == 0):
        return np.zeros((1, n))
    devs = np.zeros((1, n))
    devs[0][0] = 1
    devs_next = np.empty((0, n))
    for i in range(A - 1):
        devs_next = next_divide(devs)
        devs = devs_next
    return (devs)

def symmetrized_pure_payoff_a(x_a, x_b):
  tmp_b = np.array(list(set(permutations(x_b.tolist()))))
  signum = (np.sign((tmp_b - x_a)).sum(axis=1))
  pure_payoffs = np.sign(signum)
  payoff = pure_payoffs.sum() / tmp_b.shape[0]
  return - payoff

def best_symmetrized_respone_a(x_b, set_A):
  index = 0
  value = -1
  for i in range(set_A.shape[0]):
    if(-symmetrized_pure_payoff_a(x_b, set_A[i]) > value):
      index = i
      value =  - symmetrized_pure_payoff_a(x_b, set_A[i])
  # if(value == -1): # TODO coś mądrego tu trzeba wymyślić
  #   raise NameError('Strategia ściśle zdominowana')
  return set_A[index]

def all_best_symmetrized_respone(x_b, set_A):
  n = x_b.shape[0]
  best_responses = np.empty((0, n))
  value = -1
  # print("range: ", set_A.shape[0])
  for i in range(set_A.shape[0]):
    if(-symmetrized_pure_payoff_a(x_b, set_A[i]) > value):
      best_responses = np.empty((0, x_b.shape[0]))
      value =  - symmetrized_pure_payoff_a(x_b, set_A[i])
      best_responses = np.append(best_responses, set_A[i].reshape(1,n), axis=0)
    elif(-symmetrized_pure_payoff_a(x_b, set_A[i]) == value):
      best_responses = np.append(best_responses, set_A[i].reshape(1,n), axis=0)
  # if(value == -1): # TODO coś mądrego tu trzeba wymyślić
  #   raise NameError('Strategia ściśle zdominowana')
  return best_responses

def all_best_symmetrized_respone_respone(x_b, set_A, set_B):
  n = x_b.shape[0]
  responses = all_best_symmetrized_respone(x_b, set_A)
  responses_responses = np.empty((0,n))
  for i in range(responses.shape[0]):
    responses_responses = np.append(responses_responses, all_best_symmetrized_respone(responses[i], set_B), axis=0)
  return responses_responses

def find_symetrized_pure_eq(x_b, set_A, set_B):
  n = x_b.shape[0]
  responses = all_best_symmetrized_respone(x_b, set_A)
  A_eqs = np.empty((0,n))
  B_eqs = np.empty((0,n))
  for i in range(responses.shape[0]):
    if(x_b.tolist() in all_best_symmetrized_respone(responses[i], set_B).tolist()):
      B_eqs = np.append(B_eqs, x_b.reshape(1,n), axis=0)
      A_eqs = np.append(A_eqs, responses[i].reshape(1,n), axis=0)
  return B_eqs, A_eqs

def payoff_matrix(A, B, n):
  A_symmetrized_strategies = divides(A, n)
  B_symmetrized_strategies = divides(B, n)
  matrix = np.zeros((A_symmetrized_strategies.shape[0], B_symmetrized_strategies.shape[0]))
  for i in range(A_symmetrized_strategies.shape[0]):
    for j in range (B_symmetrized_strategies.shape[0]):
      matrix[i,j] = symmetrized_pure_payoff_a(A_symmetrized_strategies[i], B_symmetrized_strategies[j])
  return matrix

def pd_payoff_matrix(A, B, n):
  A_symmetrized_strategies = divides(A, n)
  B_symmetrized_strategies = divides(B, n)
  matrix = np.zeros((A_symmetrized_strategies.shape[0], B_symmetrized_strategies.shape[0]))
  columns_names = []
  rows_names = []
  for i in range(A_symmetrized_strategies.shape[0]):
    rows_names.append(str(A_symmetrized_strategies[i]))
  for i in range(B_symmetrized_strategies.shape[0]):
    columns_names.append(str(B_symmetrized_strategies[i]))
  for i in range(A_symmetrized_strategies.shape[0]):
    for j in range (B_symmetrized_strategies.shape[0]):
      matrix[i,j] = symmetrized_pure_payoff_a(A_symmetrized_strategies[i], B_symmetrized_strategies[j])
  df = pd.DataFrame(matrix, columns = columns_names, index=rows_names)
  return df

def isEqulibrium(strategy_A, strategy_B, payoff_mat): # TODO isEq
    holder_A = 1*(strategy_A>0)
    holder_b = 1*(strategy_B>0)

def dominating_row(row_1, row_2):
  for i in range(row_1.shape[0]):
    if(row_2[i] > row_1[i]):
      return False
  return True

def dominating_column(column_1, column_2):
  for i in range(column_1.shape[0]):
    if(column_1[i] > column_2[i]):
      return False
  return True

def remove_dominated_startegies_row_player(df):
  for i in range(df.shape[0]):
    for j in range(i + 1, df.shape[0]):
      # print(j)
      if(dominating_row(df.iloc[i], df.iloc[j])):
        df = df.drop(df.index[j])
        # print(df)
        return remove_dominated_startegies_row_player(df)
  return df

def try_reading_symmetric_matrix(resources_number, fields_number):
    try:
        path = "./data/payoff_matrices/payoff_matrix(" + str(resources_number) + \
               "," + str(resources_number) + "," + str(fields_number) + ").csv"
        payoff_mat = pd.read_csv(path, index_col=0)
        payoff_mat *= -1
    except:
        print("Loaded failed")
    return payoff_mat

print(pd_payoff_matrix(6,6,5))
# print(remove_dominated_startegies_row_player(payoff_matrix(6,6,5)))
print(try_reading_symmetric_matrix(6,5))
# np.delete(pd.read_csv(path).to_numpy(), 0,1)