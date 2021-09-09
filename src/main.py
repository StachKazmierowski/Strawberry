from mwu import MWU_game_algorithm
from solutions_evaluator import find_marginal_distribution, get_strategies, epsilon_value
from payoff_object import find_and_save_matrix
from utils import try_reading_matrix_numpy, RESULTS_PATH, PHI, STEPS_NUMBER
import os
from os import path
import datetime
import time
import sys


def run_experiment(A,B,n,steps_number,phi):
    payoff_mat = try_reading_matrix_numpy(A,B,n)
    start_time = time.time()
    results = MWU_game_algorithm(payoff_mat, phi, steps_number)
    delta_time = time.time() - start_time
    marginal_distribution_A, marginal_distribution_B = find_marginal_distribution(A,n,results[0]), find_marginal_distribution(B,n,results[1])
    game_value_A, game_value_B = results[2], results[3]
    strategy_A, strategy_B = get_strategies(A,B,n,results)
    epsilon_A, epsilon_B = epsilon_value(results[0], results[1], payoff_mat)
    dir_path_fields = RESULTS_PATH + str(n) + "_fields"
    dir_path_precise = dir_path_fields + "/results" + "(" + str(A) + "," + str(B) + "," + str(n) + ")"
    if (not path.exists(dir_path_fields)):
        os.makedirs(dir_path_fields)
    if (not path.exists(dir_path_precise)):
        os.makedirs(dir_path_precise)
    marginal_distribution_A.to_csv(dir_path_precise + "/A_marginal_distribution.csv", header=None)
    strategy_A.to_csv(dir_path_precise + "/A_strategy.csv", header=None)
    f = open(dir_path_precise + "/A_value.txt", "w")
    f.write(str(game_value_A))
    f.close()
    f = open(dir_path_precise + "/A_epsilon_value.txt", "w")
    f.write(str(epsilon_A))
    f.close()

    marginal_distribution_B.to_csv(dir_path_precise + "/B_marginal_distribution.csv", header=None)
    strategy_B.to_csv(dir_path_precise + "/B_strategy.csv", header=None)
    f = open(dir_path_precise + "/B_value.txt", "w")
    f.write(str(game_value_B))
    f.close()
    f = open(dir_path_precise + "/B_epsilon_value.txt", "w")
    f.write(str(epsilon_B))
    f.close()

    f = open(dir_path_precise + "/raport.txt", "w")
    now = datetime.datetime.now()
    f.write("Przybliżone równowagi wyznaczono: " + str(now) + "\n")
    f.write("Parametry algorytmu:\n")
    f.write("STEPS_NUMBER = " + str(steps_number) + "\n")
    f.write("PHI = " + str(phi) + "\n")
    f.write("Czas działania algorytmu: " + str(delta_time) + "s\n")
    f.close()

if __name__ == "__main__":
    args = sys.argv
    if(len(args) > 7 or len(args) < 4):
        print("NIEPRAWIDŁOWA LICZBA ARGUMENTÓW")
        print("Podaj nastąpujące trzy argumenty: zasoby_A, zasoby_B, liczba pól")
        print("Możesz podać opcjonalnie następne trzy argumenty (domyślne wartości): liczba_kroków(10000), parametr_phi(1/8), liczba wątków przy obliczaniu macierzy (maksymalna)")
    else:
        A,B,n = int(args[1]), int(args[2]), int(args[3])
        if (len(args) > 4):
            steps_number = int(args[4])
        else:
            steps_number = STEPS_NUMBER
        if (len(args) > 5):
            phi = float(args[5])
        else:
            phi = PHI
        if (len(args) > 6):
            threads_num = int(args[6])
        else:
            threads_num = None
        find_and_save_matrix(A, B, n, threads_num)
        run_experiment(A,B,n,steps_number,phi)