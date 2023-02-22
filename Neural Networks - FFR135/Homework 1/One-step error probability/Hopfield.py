# Artificial Neural Network  Homework 1, Nicole Adamah
# Library for one-step error probability
import numpy as np

def pattern_Generator(patterns, N):
    return np.random.choice([-1,1], size=(patterns, N))

def sgn(x):
    return -1 if x < 0 else 1

class Hopfield:
    def __init__(self, patterns, N, zerodiagonal:bool):
        self.N = N
        self.zerodiagonal = zerodiagonal
        self.patterns = pattern_Generator(patterns, N)
        self.w = np.zeros((N,N))
        self.hebbs_rule()

    def hebbs_rule(self):

        for x in self.patterns:
            o = x.reshape(self.N,1)
            ot = np.transpose(o)
            self.w += np.matmul(o, ot)# Matrix product of two arrays
            if self.zerodiagonal == True:
                np.fill_diagonal(self.w, 0)
        return self.w / self.N

    def asynchronous(self, state, i):
        self.neuron_weight = self.w[i, :]
        b = np.matmul(self.neuron_weight, state)
        signum = sgn(b)
        if signum == 0:
            signum = 1
        return signum
