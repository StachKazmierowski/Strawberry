import re
import numpy as np
import pandas as pd

path = './data/'
filename = 'tmp.txt'

def read_strategy_line(line):
    strategy = re.search('{(.+?)}', line).group(1)
    resources = strategy.split(',')
    resources = [float(i) for i in resources]
    resources.sort(reverse=True)
    resources = np.array(resources)
    line = line[0:-2]
    number_of_usages = int(line.split(',')[-1])
    probability_frac = line.split(',')[-2].split('/')
    probability_numerator = int(probability_frac[0])
    probability_denomirator = int(probability_frac[1])
    probability = number_of_usages * probability_numerator/probability_denomirator
    return str(resources), probability

def parse_file(path, filename):
    with open (path + filename, "r") as myfile:
        out = []
        data=myfile.readlines()
        # print(data)
        # print(data[0])
        for i in range(len(data)):
            out.append(read_strategy_line(data[i]))
        out = np.array(out, dtype=object)
        out = pd.DataFrame(out[:,1], index=out[:,0]).sort_index(ascending=False)
        return out

print(parse_file(path, filename))

