from symmetrized.utils import divides, all_best_symmetrized_respone, all_best_symmetrized_respone_respone, np


def find_number_of_symetric_eq(A, n):
    A_strategies = divides(A, n)
    A_optimal = np.empty((0, n))
    for i in range(A_strategies.shape[0]):
        if (A_strategies[i].tolist() in all_best_symmetrized_respone(A_strategies[i], A_strategies).tolist()):
            A_optimal = np.unique(np.append(A_optimal, A_strategies[i].reshape(1, n), axis=0), axis=0)
    return A_optimal.shape[0]

def find_symetric_eq(A, n):
    A_strategies = divides(A, n)
    A_optimal = np.empty((0, n))
    for i in range(A_strategies.shape[0]):
        if (A_strategies[i].tolist() in all_best_symmetrized_respone(A_strategies[i], A_strategies).tolist()):
            A_optimal = np.unique(np.append(A_optimal, A_strategies[i].reshape(1, n), axis=0), axis=0)
    return A_optimal

def find_number_of_eq(A, B, n):
    A_strategies = divides(A, n)
    B_strategies = divides(B, n)
    A_optimal = np.empty((0, n))
    for i in range(A_strategies.shape[0]):
        for k in range(all_best_symmetrized_respone_respone(A_strategies[i], B_strategies, A_strategies).tolist().count(
                A_strategies[i].tolist())):
            A_optimal = np.append(A_optimal, A_strategies[i].reshape(1, n), axis=0)
    return A_optimal.shape[0]

def has_pure_eq(A, B, n):
    A_strategies = divides(A, n)
    B_strategies = divides(B, n)
    for i in range(A_strategies.shape[0]):
        for k in range(all_best_symmetrized_respone_respone(A_strategies[i], B_strategies, A_strategies).tolist().count(
                A_strategies[i].tolist())):
            return True
    return False


# print(find_number_of_eq(9, 4, 3))

# find_number_of_symetric_eq(9, 3)
# %%
import pandas as pd
import time

FIELDS_MAX = 11
RES_MAX = 11
no_eq = np.empty((0, 2))

for fields in range(2, FIELDS_MAX):
    for resources in range(1, RES_MAX):
        if(resources > fields):
            start_time = time.time()
            tmp = has_pure_eq(resources, resources, fields)
            print(fields, resources, time.time() - start_time)
            if (not tmp):
                no_eq = np.append(no_eq, np.array([resources, fields]).reshape(1, 2), axis=0)
                df = pd.DataFrame(no_eq, columns=["Resources", "Fields"], dtype="int8")
                df.to_csv("./data/no_pure_equilibria.csv", index=False)

df.to_csv("./data/no_pure_equilibria_final.csv", index=False)
#%%
import matplotlib.pyplot as plt
df = pd.read_csv("./data/no_pure_equilibria.csv")
numbers = np.zeros((10, 10))
for i in range(df.shape[0]):
    numbers[df.iloc[i][0] - 1, df.iloc[i][1] - 1] = 1

def prettyPrint(harvest, title):
    input = list(range(1, 11))
    output = list(range(1, 11))
    print(input)
    fig, ax = plt.subplots()
    im = ax.imshow(harvest)

    # We want to show all ticks...
    ax.set_xticks(np.arange(harvest.shape[1]))
    ax.set_yticks(np.arange(harvest.shape[0]))
    # ... and label them with the respective list entries
    ax.set_xticklabels(input)
    ax.set_yticklabels(output)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(harvest.shape[0]):
        for j in range(harvest.shape[1]):
            text = ax.text(j, i, f"{harvest[i, j]:.3f}",
                           ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    fig.set_size_inches(20, 20)
    fig.savefig(title + '.png', dpi=200,format='png')
    plt.show()

print(numbers.shape)
prettyPrint(numbers, "")
print(numbers)
# print(df)

#%%
def has_pure_eq_wnt_max(B, n):
    A_strategies = np.ones((1,n))
    A_strategies[-n:-1] = 0
    A_optimal = np.empty((0, n))
    for i in range(A_strategies.shape[0]):
        if (A_strategies[i].tolist() in all_best_symmetrized_respone(A_strategies[i], A_strategies).tolist()):
            return True
    return False

MAX = 11
for i in range(2, MAX):
    for j in range(2, MAX):
        if(i < j):
            print(i, j, has_pure_eq_wnt_max(i,j))

