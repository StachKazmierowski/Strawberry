import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
fields_MIN = 3
fields_MAX = 10
res_MIN = 10
res_MAX = 25
tmp = pd.read_csv("./res/symmetric/times/20_20_10.csv", index_col=0)
print(tmp)
print(list(tmp.iloc[0]))
print(list(tmp.columns))
# print(tmp[tmp.columns[0]])

#%%

def plot_1(phi_exponent):
    plt.clf()
    for i in range(10,22, 2):
        tmp_label = str(i) + " zasobów"
        filename = "./res/symmetric/times/" + str(i) + "_" + str(i) + "_10.csv"
        tmp = pd.read_csv(filename, index_col=0)
        plt.scatter(list(tmp.columns.astype(int)), list(tmp.iloc[phi_exponent-1]), label=tmp_label)
        plt.xlabel("Liczba kroków")
        plt.ylabel("Czas[s]")
        plt.xscale('log')
        plt.yscale('log')

    plt.title("Czasu działania algorytmu dla phi=" + '$(0.5)^' + str(phi_exponent) + '$')
    plt.legend()
    plt.savefig("./plots_mwu/time_steps" + str(phi_exponent) + "=phi.png")
    plt.show()
for i in range(2,6):
    plot_1(i)
#%%
for phi_exponent in range (1, 6):
    # plt.clf()
    for i in range(10,21, 2):
        tmp_label = str(phi_exponent) + " wykładnik"
        filename = "./res/symmetric/times/" + str(i) + "_" + str(i) + "_10.csv"
        tmp = pd.read_csv(filename, index_col=0)
        plt.plot(list(tmp.columns.astype(int)), list(tmp.iloc[phi_exponent]), label=tmp_label)
        plt.xlabel("Liczba kroków")
        plt.ylabel("Czas[s]")
        # plt.xscale('log')
        # plt.yscale('log')

plt.title("Iloraz czasu działania algorytmu dla phi=" + '$(0.5)^' + str(phi_exponent) + '$')
plt.legend()
# plt.savefig("./plots/time(fields)_plt.png")
plt.show()
#%%
def plot_epsilon_steps_number(phi_exponent, fields_num, res_num):
    # phi_exponent = 5
    plt.clf()
    for i in range(res_num,res_num+1, 2):
        tmp_label = "strategia gracza kolumnowego"
        filename = "./res/symmetric/col_col/" + str(i) + "_" + str(i) + "_" + str(fields_num) + ".csv"
        tmp = pd.read_csv(filename, index_col=0)
        plt.plot(list(tmp.columns.astype(int)), list(tmp.iloc[phi_exponent-1]), label=tmp_label)

        # tmp_label = "Strategia kolumnowa z wierszową"
        # filename = "./res/symmetric/row_col/" + str(i) + "_" + str(i) + "_" + str(fields_num) + ".csv"
        # tmp = pd.read_csv(filename, index_col=0)
        # plt.plot(list(tmp.columns.astype(int)), list(tmp.iloc[phi_exponent-1]), label=tmp_label)

        tmp_label = "strategia gracza wierszowego"
        filename = "./res/symmetric/row_row/" + str(i) + "_" + str(i) + "_" + str(fields_num) + ".csv"
        tmp = pd.read_csv(filename, index_col=0)
        plt.plot(list(tmp.columns.astype(int)), list(tmp.iloc[phi_exponent-1]), label=tmp_label)

        plt.xlabel("Liczba kroków")
        plt.ylabel("epsilon")
        # plt.xscale('log')
        plt.yscale('log')

    plt.title("Uzyskane dokładności dla A=B=" + str(res_num) + ", n=" + str(fields_num) + ", phi = $(0.5)^" + str(phi_exponent) + "$")
    plt.legend()
    # plt.show()
    plt.savefig("./plots_mwu/epsilon_row_col_" + str(res_num) + "_" + str(res_num) + "_" + str(fields_num) + ".png")
#
plot_epsilon_steps_number(5, 7, 19)

#%%
# phi_exponent = 5
def phis():
    ret = []
    for i in range(1, 11):
        ret.append(1/(2**i))
    return ret

print(phis())
print([(2**(-7+i)) for i in range (6)])

def plot_epsilon_phi(steps_num, fields_num):
    plt.clf()
    for res_num in range(15,22,2):
        fields_num = 10
        tmp_label = "liczba zasobów = " + str(res_num)
        filename = "./res/symmetric/col_col/" + str(res_num) + "_" + str(res_num) + "_" + str(fields_num) + ".csv"
        tmp = pd.read_csv(filename, index_col=0)
        plt.scatter(phis()[1:-3], list(tmp[str(steps_num)])[1:-3], label=tmp_label)
    plt.xlabel("phi")
    plt.ylabel("epsilon")
    plt.semilogx(base=2)
    # plt.yscale('log')

    plt.title("Uzyskane dokładności dla n=" + str(fields_num) + ", liczba kroków=" + str(steps_num))
    plt.legend()
    plt.savefig("./plots_mwu/lin_epsilon_phi" + str(fields_num) + "_fields" + str(steps_num) + "_steps.png")
    plt.show()

plot_epsilon_phi(int(32768/2), 10)
print(phis().reverse())
#%%
phi_exponent = 5
fields_num = 10
def plot_list(fields_num):
    tmp = []
    for i in range(10,22):
        file = pd.read_csv("./res/symmetric/times/" + str(i) + "_" + str(i) + "_" + str(fields_num) + ".csv", index_col=0)
        tmp.append(file)
    times = []
    for i in range (10,22):
        times.append(tmp[i-10].iloc[phi_exponent][-1])
    return times

plt.clf()
for i in range(4,11,2):
    tmp_label = str(i) + " pól"
    plt.scatter(list(range(10,22)), plot_list(i), label=tmp_label)
    plt.xlabel("Liczba zasobów")
    plt.ylabel("Czas[s]")
    # plt.xscale('log')
    plt.yscale('log')

plt.title("Czasu działania algorytmu dla n=10, \n phi=" + '$(0.5)^' + str(phi_exponent) + '$' + ", steps_number=32768")
plt.legend()
plt.savefig("./plots_mwu/time_fixed_fields_res_arg_" + str(phi_exponent) + "=phi.png")
plt.show()