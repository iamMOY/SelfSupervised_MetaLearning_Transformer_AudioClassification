import numpy as np


#Read a numpy file
def read_npy_file(filename):
    return np.load(filename)

x = read_npy_file('1-11687-A-47.npy')
print(x.shape)

print()