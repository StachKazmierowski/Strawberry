import time
from symmetrized.mwu import MWU_game_algorithm
start_time = time.time()
print(time.time() - start_time)

for i in range(2, 3):
    start_time = time.time()
    print("i", i)
    print(MWU_game_algorithm(i, i,i))
    print("czas", time.time() - start_time)


#%%
start_time = time.time()
print(MWU_game_algorithm(10,10, 10, 1/10, 10))
print("czas", time.time() - start_time)

#%%
import pandas as pd
import numpy as np
fields = list(range(1,10))
resources = list(range(1,10))

def run_sigle(fields, resources, phi, step_number, power):
    small = 10**(-power)
    times = np.zeros((len(resources),len(fields)))
    percent_size = np.zeros((len(resources),len(fields)))
    for i in range(len(fields)):
        for j in range(len(resources)):
            start_time = time.time()
            # tmp = MWU_carrier_percent_size(resources[j], fields[i], phi, step_number, small)
            # percent_size[j,i] = tmp
            times[j,i] = time.time() - start_time
    df_times = pd.DataFrame(np.around(times, decimals=power), index=resources, columns=fields)
    df_percent_size = pd.DataFrame(np.around(percent_size, decimals=power),  index=resources, columns=fields)
    df_times.to_csv("./results/times/phi=" + str(phi) + "_SN=" + str(step_number) + "_small=" + str(small) + ".csv")
    df_percent_size.to_csv("./results/percent_size/phi=" + str(phi) + "_SN=" + str(step_number) + "_small=" + str(small) + ".csv")

step_sizes = [10,20,40]
phis = [3/4,1/2,1/4]
powers = [3,5,7]
#%%
for step_size in step_sizes:
    for phi in phis:
        for power in powers:
            print("step_size", step_size, "phi", phi, "power", power)
            run_sigle(fields, resources, phi, step_size, power)
#%%
print(list(range(1,20)))

run_sigle([1],[1], 1/2, 20, 3)
