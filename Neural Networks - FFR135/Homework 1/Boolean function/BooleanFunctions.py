# Artificial Neural Network  Homework 1, Nicole Adamah
# Calculating the number of linear separable boolean functions for n-dimensions
import numpy as np
import itertools
from tqdm import tqdm
trials = 10**4
epochs = 20
eta = 0.05

N = 2 # change n to the desired dimension
counter = 0
variance = 1/N
used_bool = []
x_innt = []
# initializing the input arrays
input = list(itertools.product([-1, 1], repeat=N))
for x in input:
    x_innt.append(x)
x_innt = np.array(x_innt)

def sgn(x):
    return -1 if x < 0 else 1

def learning_rule(x, w, theta):
    b = (np.matmul(x,w)) - theta
    return b


for trial in tqdm(range(trials)):
    bool_f = np.random.choice([-1, 1], size=(2 ** N), p=[1/2,1/2])
    bool_f = bool_f.tolist()
    if bool_f not in used_bool:
        a = np.sqrt(((12) * variance)) / 2
        w = np.random.uniform(-a, a, size=N)
        theta = 0
        for epoch in range(0, epochs):
            total_error = 0
            for mu in range(0,2**N):
                used_bool.append(bool_f)
                pattern = x_innt[mu,:]
                b = learning_rule(pattern, w, theta)
                outputs = sgn(b)
                error = (bool_f[mu]) - (outputs)
                w += eta * error * pattern
                theta -= eta * error
                total_error += abs(error)

            if total_error ==  0:
                counter += 1
                break



print(counter)



