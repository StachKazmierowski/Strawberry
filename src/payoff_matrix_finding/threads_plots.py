import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
fields_MIN = 5
fields_MAX = 16
res_MIN = 5
res_MAX = 16
threads_MIN = 1
threads_MAX = 9
#%%


plt.clf()
filename = "./data/threads_times/time_threads_fields.csv"
tmp = pd.read_csv(filename, index_col=0)
print(tmp)
print(tmp["5"])
#%%
filename = "./data/threads_times/time_threads_fields.csv"
tmp = pd.read_csv(filename, index_col=0)
# print(tm)
for fields in range(fields_MIN,fields_MAX, 2):
    t_0 = list(tmp[str(fields)])[0]
    tmp_list = [t_0/x for x in tmp[str(fields)]]
    tmp_label = str(fields) + " pól"
    plt.plot(list(tmp.index.astype(int)), tmp_list, label=tmp_label)
    plt.xlabel("Liczba wątków")
    plt.ylabel("Przyspieszenie")
    # plt.xscale('log', base=2)
    # plt.yscale('log',  base=2)

plt.title("Przyspieszenie algorytmu dla wielu wątków, \n wyniki dla symetrycznych konfliktów, liczba zasobów to 10.")
plt.legend()
plt.savefig("./plots/threads_speed_up_fixed_res.png")
plt.show()
# for i in range(2,6):
#     plot_1(i)
#%%
filename = "./data/threads_times/time_threads_res.csv"
tmp = pd.read_csv(filename, index_col=0)

for res in range(res_MIN,res_MAX, 2):
    t_0 = list(tmp[str(res)])[0]
    tmp_list = [t_0/x for x in tmp[str(res)]]
    tmp_label = str(res) + " zasobów"
    plt.plot(list(tmp.index.astype(int)), tmp_list, label=tmp_label)
    plt.xlabel("Liczba wątków")
    plt.ylabel("Przyspieszenie")
    # plt.xscale('log', base=2)
    # plt.yscale('log',  base=2)

plt.title("Przyspieszenie algorytmu dla wielu wątków, \n wyniki dla symetrycznych konfliktów, liczba pól to 10.")
plt.legend()
plt.savefig("./plots/threads_speed_up_fixed_fields.png")
plt.show()
#%%
xs = [0,1,2,3,4,5]
ys = [2**(2**i) for i in xs]
print(ys)
plt.plot(ys)
plt.xscale('log', base=2)
plt.yscale('log',  base=2)
plt.show()
