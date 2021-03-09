from utils import payoff_matrix, pd_payoff_matrix, divides
import numpy as np
import pandas as pd
np.set_printoptions(precision=10, suppress=True)

def MWU_symmetric_game_algorithm_working(resources_number, fields_number, phi, steps_number):
    payoff_mat = payoff_matrix(resources_number, resources_number, fields_number)
    n = payoff_mat.shape[0]
    # print(payoff_mat)
    TAIL_FOR_AVG_SIZE = steps_number/2
    strategy = np.ones((n, 1))
    strategy = strategy/n
    strategy_sum = np.zeros_like(strategy)
    j = 1
    for i in range (steps_number):
        # print(steps_number - TAIL_FOR_AVG_SIZE)
        payoffs = np.matmul(payoff_mat, strategy)
        strategy = np.multiply(( (phi -1) * (payoffs <= -0.01)+1), strategy)
        strategy = strategy/strategy.sum()
        strategy[np.argsort(payoffs.reshape(n,).tolist())[-1]] += 0.002
        strategy = strategy/strategy.sum()
        # print(strategy.sum())
        if(i > steps_number - TAIL_FOR_AVG_SIZE):
            # print("tera")
            strategy_sum += strategy
        if(i > 10**j):
            # print(phi)
            phi = (0.99+phi)/2
    return strategy_sum/strategy_sum.sum()
res = MWU_symmetric_game_algorithm_working(6,5,0.01,4000)
print(np.matmul(payoff_matrix(6,6,5), res))
print(res)
#%%
res = MWU_symmetric_game_algorithm_working(7,5,0.5,60000)
print(res)
print(np.matmul(payoff_matrix(7,7,5), res))
# print(res >= res[np.argsort(res.reshape(10,).tolist())[-10]])
#%%
print(divides(6,5))

