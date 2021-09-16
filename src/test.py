import numpy as np
from basic_utils import payoff_matrix_more_than_opponent, payoff_matrix_more_than_half, payoff_matrix_more_than_half_pd
from payoff_object import payoff_dynamic_finder, payoff_matrix_pd
from payoff_more_than_half_object import payoff_dynamic_finder_more_than_half, payoff_matrix_pd

## Test for more than opponent

# finder = payoff_dynamic_finder()
# # print((finder.payoff_matrix(9,9,3) - payoff_matrix_more_than_opponent(9,9,3) == 0).all())
# errors = 0
# for n in range (1, 6):
#     print("current n", n)
#     for A in range(3, 15):
#         print("current A", A)
#         for B in range(3, 16):
#             if(not (finder.payoff_matrix(A,B,n) - payoff_matrix_more_than_opponent(A,B,n) == 0).all()):
#                 print(A,B,n)
#                 errors += 1
# print("errors found", errors)

# test for more than half
finder = payoff_dynamic_finder_more_than_half()
# print((finder.payoff_matrix(9,9,3) - payoff_matrix_more_than_opponent(9,9,3) == 0).all())
errors = 0
for n in range (1, 5, 2):
    print("current n", n)
    for A in range(3, 15):
        print("current A", A)
        for B in range(3, 16):
            if(not (finder.payoff_matrix(A,B,n) - payoff_matrix_more_than_half(A,B,n) == 0).all()):
                print(A,B,n)
                errors += 1
print("errors found", errors)

# finder = payoff_dynamic_finder_more_than_half()
# print(payoff_matrix_pd(finder.payoff_matrix(7,5,3), 7, 5, 3) )
# print("================================")
# print(payoff_matrix_more_than_half_pd(7,5,3))