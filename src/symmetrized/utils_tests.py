import unittest
from src.symmetrized.utils import divides, next_divide, symmetrized_pure_payoff_a, payoff_matrix, pd_payoff_matrix

# TODO trzeba zrobić testy
class MyTestCase(unittest.TestCase):
    def test_divides_generating(self):
        for i in range(2, 10):
            self.assertTrue((next_divide(divides(20, i)) ==  divides(21, i)).all())

    def test_symmetrized_payoff(self):
        symmetrized_strategies_A = divides(7,3)
        symmetrized_strategies_B = divides(10,3)
        for i in range(symmetrized_strategies_A.shape[1]):
            strategy_A = symmetrized_strategies_A[:,i]
            for j in range(symmetrized_strategies_B.shape[1]):
                strategy_B = symmetrized_strategies_A[:,i]
                self.assertEqual(symmetrized_pure_payoff_a(strategy_A, strategy_B),
                                 -symmetrized_pure_payoff_a(strategy_B, strategy_A))

    def test_divides_generating(self):
        for fields in range(2, 5):
            for i in range(2, 6):
                for j in range(2, 6):
                    self.assertTrue((pd_payoff_matrix(i, j, fields).to_numpy() ==  payoff_matrix(i,j,fields)).all())

if __name__ == '__main__':
    unittest.main()


