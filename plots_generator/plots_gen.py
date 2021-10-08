import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
fields_MIN = 3
fields_MAX = 10
res_MIN = 10
res_MAX = 25
tmp = pd.read_csv("./experiments_results/times/(20,20,11).csv", index_col=0)
print(tmp)
#%%
def phis():
    ret = []
    for i in range(1, 11):
        ret.append(1/(2**i))
    return ret

print(phis())

def step_numbers():
    ret = []
    for i in range(14):
        ret.append(125*(2**i))
    return ret

print(step_numbers())

def plot_time_steps():
    tmp = pd.read_csv("./experiments_results/times/(25,25,9).csv", index_col=0)
    times = list(tmp.iloc[0])
    print(times)
    plt.plot(list(tmp.columns.astype(int)), times)
    print(list(tmp.columns.astype(int)))
    plt.show()

plot_time_steps()
#%%
def plot_1(phi_exponent, n):
    plt.clf()
    for i in range(15,31,3):
        tmp_label = str(i) + " resources"
        filename = "./experiments_results/times/(" + str(i) + "," + str(i) + "," + str(n) + ").csv"
        tmp = pd.read_csv(filename, index_col=0)
        plt.plot(list(tmp.columns.astype(int)), list(tmp.iloc[phi_exponent-1]), label=tmp_label)
        plt.xlabel("Number of steps")
        plt.ylabel("Time[s]")

    # plt.title("Algorithm runtime for phi=" + '$(0.5)^' + str(phi_exponent) + '$, ' + str(n) + ' battlefields')
    plt.legend()
    # plt.semilogx(base=2)
    # plt.subplots()[1].ticklabel_format(style='plain')
    plt.ticklabel_format(style='plain')
    plt.savefig("./plots_mwu/time_steps" + str(phi_exponent) + "=phi.png")
    plt.show()
for i in range(2,6):
    plot_1(i, 15)
# plot_1(4)
#%%
def plot_epsilon_steps_number(phi_exponent, fields_num, res_num):
    # phi_exponent = 5
    plt.clf()
    for i in range(res_num,res_num+1, 2):
        tmp_label = "pair of column player strategies"
        filename = "./experiments_results/col_col/(" + str(i) + "," + str(i) + "," + str(fields_num) + ").csv"
        tmp = pd.read_csv(filename, index_col=0)
        plt.plot(list(tmp.columns.astype(int)), list(tmp.iloc[phi_exponent-1]), label=tmp_label)

        tmp_label = "column and row players strategies"
        filename = "./experiments_results/row_col/(" + str(i) + "," + str(i) + "," + str(fields_num) + ").csv"
        tmp = pd.read_csv(filename, index_col=0)
        plt.plot(list(tmp.columns.astype(int)), list(tmp.iloc[phi_exponent-1]), label=tmp_label)

        tmp_label = "pair of row player strategies"
        filename = "./experiments_results/row_row/(" + str(i) + "," + str(i) + "," + str(fields_num) + ").csv"
        tmp = pd.read_csv(filename, index_col=0)
        plt.plot(list(tmp.columns.astype(int)), list(tmp.iloc[phi_exponent-1]), label=tmp_label)

        plt.xlabel("Number of steps")
        plt.ylabel("epsilon")
        plt.xscale('log')
        plt.yscale('log')

    # plt.title("Accuracies for " + str(res_num) + " resources , " + str(fields_num) + " battlefields, phi=$(0.5)^" + str(phi_exponent) + "$")
    plt.legend()
    plt.savefig("./plots_mwu/epsilon_row_col_" + str(res_num) + "_" + str(res_num) + "_" + str(fields_num) + ".png")
    plt.show()
plot_epsilon_steps_number(7, 15, 29)

#%%
# phi_exponent = 5

# print([(2**(-7+i)) for i in range (6)])

def plot_epsilon_phi(steps_num, fields_num):
    plt.clf()
    for res_num in range(17,26,2):
        tmp_label = str(res_num) + " resources"
        filename = "./experiments_results/row_row/(" + str(res_num) + "," + str(res_num) + "," + str(fields_num) + ").csv"
        tmp = pd.read_csv(filename, index_col=0)
        plt.plot(phis(), list(tmp[str(steps_num)]), label=tmp_label)
    plt.xlabel("phi")
    plt.ylabel("epsilon")
    plt.semilogx(base=2)
    plt.yscale('log')

    # plt.title("The obtained accuracies for " + str(fields_num) + " battlefields, " + str(steps_num) + " steps")
    plt.legend()
    plt.savefig("./plots_mwu/epsilon_phi" + str(fields_num) + "_fields" + str(steps_num) + "_steps.png")
    plt.show()

plot_epsilon_phi(125*2**13, 15)
#%%
phi_exponent = 6
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
    tmp_label = str(i) + " fields"
    plt.scatter(list(range(10,22)), plot_list(i), label=tmp_label)
    plt.xlabel("Number of resources")
    plt.ylabel("Time[s]")
    # plt.xscale('log')
    plt.yscale('log')

plt.title("Algorithm runtime for n=10, \n phi=" + '$(0.5)^' + str(phi_exponent) + '$' + ", steps_number=32768")
plt.legend()
plt.savefig("./plots_mwu/time_fixed_fields_res_arg_" + str(phi_exponent) + "=phi.png")
plt.show()
#%%
def cp_vs_np_plot():
    n = 15
    cp_times =[]
    np_times = []
    for i in range(15, 31):
        filename_np = "./experiments_results/times/(" + str(i) + "," + str(i) + "," + str(n) + ").csv"
        filename_cp = "./experiments_results_cp/times/(" + str(i) + "," + str(i) + "," + str(n) + ").csv"
        tmp_np = pd.read_csv(filename_np, index_col=0)
        tmp_cp = pd.read_csv(filename_cp, index_col=0)
        np_times.append(list(tmp_np.iloc[0])[-1])
        cp_times.append(list(tmp_cp.iloc[0])[-1])
    plt.plot(list(range(15, 31)), np_times, label="CPU")
    plt.plot(list(range(15, 31)), cp_times, label="GPU")
    # plt.title("Algorithm runtime comparison for GPU and CPU,\n 15 battlefields, phi=" + '$(0.5)$')
    plt.xlabel("Number of resources")
    plt.ylabel("Time[s]")
    plt.legend()
    plt.savefig("./plots_mwu/cupy_vs_numpy.png")
    # plt.yscale('log')
    plt.show()

cp_vs_np_plot()
#%%
plt.clf()
sizes_15 = [176, 230, 295, 381, 483, 615, 773, 972, 1210, 1508, 1861, 2297, 2815, 3446, 4192, 5096]
sizes_17 = [176, 231, 297, 384, 488, 623, 785, 990, 1236, 1545, 1913, 2369, 2913, 3579, 4370, 5332]
sizes_19 = [176, 231, 297, 385, 490, 626, 790, 998, 1248, 1563, 1939, 2406, 2965, 3651, 4468, 5465]
sizes_21 = [176, 231, 297, 385, 490, 627, 792, 1001, 1253, 1571, 1951, 2424, 2991, 3688, 4520, 5537]
plt.plot(list(range(15,31)), sizes_15, label="15 fields")
# plt.plot(list(range(15,31)), sizes_17, label="17 fields")
# plt.plot(list(range(15,31)), sizes_19, label="19 fields")
plt.plot(list(range(15,31)), sizes_21, label="21 fields")
plt.xlabel("Number of resources")
plt.ylabel("Number of symmetric strategies")
# plt.yscale('log')
# plt.title("Number of symmetric strategies for 15 battlefields")
plt.legend()
plt.savefig("./plots_mwu/number_of_strategies_15_fields.png")
plt.show()
#%%
def sizes(fields_number):
    result = []
    for res in range(15, 26):
        mat = pd.read_csv("./payoff_matrices/"+ str(fields_number) + "_fields/payoff_matrix(" + str(res) + "," +
                          str(res) + "," + str(fields_number) + ").csv", index_col=0)
        result.append(mat.shape[0])
    return result
plt.clf()
for fields_number in range(9,22,2):
    plt.plot(list(range(15,26)), sizes(fields_number), label=str(fields_number) + "battlefields")
plt.legend()
plt.xlabel("Number of resources")
plt.ylabel("Number of symmetric strategies")
plt.title("Number of symmetric strategies for 15 battlefields")
plt.yscale('log')
plt.show()
