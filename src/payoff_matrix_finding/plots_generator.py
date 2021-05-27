import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
fields_MIN = 3
fields_MAX = 10
res_MIN = 10
res_MAX = 25
tmp = pd.read_csv("./data/times/time_iter.csv", index_col=0)
print(tmp.iloc[0])
print(tmp[tmp.columns[0]])

#%%
def prepare_fields():
    fields = []
    for i in range(fields_MIN, fields_MAX + 1):
        fields.append(i)
    return fields

def prepare_reses():
    reses = []
    for i in range(res_MIN, res_MAX + 1):
        reses.append(i)
    return reses

def div_times(times):
    times_ret = []
    for j in range(len(times) - 1):
        if(times[j] == 0):
            times_ret.append(0)
        else:
            times_ret.append(times[j+1]/times[j])
    return times_ret

plt.clf()
for i in range(0, res_MAX - res_MIN +1, 5):
    tmp_label=str(i+res_MIN) + " zasobów"
    plt.plot(prepare_fields(), list(tmp.iloc[i]), label=tmp_label)
    plt.xlabel("Liczba pól")
    plt.ylabel("Czas[s]")
    plt.yscale('log')
    # plt.show()
plt.title("Czas obliczania macierzy wypłat")
plt.legend()
plt.savefig("./plots/time(fields)_plt.png")
plt.show()
#%%
tmp = pd.read_csv("./data/times/time_iter.csv", index_col=0)
plt.clf()
for i in range(0, res_MAX - res_MIN +1, 5):
    tmp_label=str(i+res_MIN) + " zasobów"
    plt.scatter(prepare_fields()[0:-1], div_times(list(tmp.iloc[i])), label=tmp_label)
    plt.xlabel("Liczba pól")
    plt.ylabel("Czas[s]")
    # plt.yscale('log')
    # plt.show()
plt.title("Iloraz czasu obliczania macierzy wypłat")
plt.legend()
plt.show()
#%%
tmp = pd.read_csv("./data/times/time_iter.csv", index_col=0)
plt.clf()
print(tmp.columns)
for i in range(3, fields_MAX-fields_MIN+1):
    # print(i)
    tmp_label=str(i+fields_MIN) + " pól"
    plt.plot(prepare_reses(), list(tmp[tmp.columns[i]]), label=tmp_label)
    plt.xlabel("Liczba zasobów")
    plt.ylabel("Czas[s]")
    plt.yscale('log')
    # plt.show()
plt.title("Czas obliczania macierzy wypłat")
plt.legend()
plt.savefig("./plots/time(res)_plt.png")
plt.show()
#%%
tmp = pd.read_csv("./data/times/cell_time_iter.csv", index_col=0)
plt.clf()
for i in range(5, res_MAX - res_MIN +1, 5):
    tmp_label=str(i+res_MIN) + " zasobów"
    plt.plot(prepare_fields()[4:], list(tmp.iloc[i]/1000)[4:], label=tmp_label)
    plt.xlabel("Liczba pól")
    plt.ylabel("Czas[ms]")
    # plt.yscale('log')
    # plt.show()
plt.title("Czas obliczania pojedyńczej wypłaty")
plt.legend()
# plt.savefig("./plots/single_time(fields).png")
plt.show()
#%%
tmp = pd.read_csv("./data/times/cell_time_iter.csv", index_col=0)
plt.clf()
print(tmp.columns)
for i in range(2, fields_MAX-fields_MIN+1):
    print(i)
    tmp_label=str(i+fields_MIN) + " pól"
    plt.scatter(prepare_reses()[0:-1], div_times(list(tmp[tmp.columns[i]]/1000)), label=tmp_label)
    plt.xlabel("Liczba pól")
    # plt.ylabel("Czas[ms]")
    # plt.yscale('log')
    # plt.show()
plt.title("Iloraz czasu obliczania pojedyńczej wypłaty")
plt.legend()
plt.show()
#%%
tmp = pd.read_csv("./data/times/cell_time_iter.csv", index_col=0)
plt.clf()
for i in range(3, fields_MAX-fields_MIN+1):
    # print(i)
    tmp_label=str(i+fields_MIN) + " pól"
    plt.scatter(prepare_reses(), list(tmp[tmp.columns[i]]/1000), label=tmp_label)
    plt.xlabel("Liczba zasobów")
    plt.ylabel("Czas[ms]")
    plt.xticks(list(range(10,26,2)))
    # plt.yscale('log')
    # plt.show()
plt.title("Czas obliczania pojedyńczej wypłaty")
plt.savefig("./plots/single_time(res).png")
plt.legend()
plt.show()
#%%
tmp = pd.read_csv("./data/times/time.csv", index_col=0)
# print(tmp)
ranges = [21,13]
ends = [18,10]
plt.clf()
for i in range(2):
    tmp_label=str(i+1) + "-krotność zasobów"
    plt.plot(list(range(3,ranges[i])), list(tmp.iloc[i])[0:ends[i]], label=tmp_label)
    plt.xlabel("Liczba pól")
    plt.ylabel("Czas[s]")
    plt.yscale('log')
    plt.xticks(list(range(3,21)))
    # plt.show()
plt.title("Czas obliczania macierzy wypłat")
plt.legend()
# plt.savefig("./plots/time_mul(fields).png")
plt.show()
#%%
tmp = pd.read_csv("./data/times/cell_time.csv", index_col=0)
# print(tmp)
plt.clf()
for i in range(2):
    tmp_label=str(i+1) + "-krotność zasobów"
    plt.scatter(list(range(3,ranges[i])), list(tmp.iloc[i]/1000)[0:ends[i]], label=tmp_label)
    plt.xlabel("Liczba pól")
    plt.ylabel("Czas[ms]")
    # plt.yscale('log')
    plt.xticks(list(range(3,21)))
    # plt.yticks(list(range(0,10,1)))
    # plt.yscale('log')
    # plt.show()
plt.title("Czas obliczania pojedyńczej wypłaty")
plt.legend()
# plt.savefig("./plots/single_time(fields)_plt.png")
plt.show()