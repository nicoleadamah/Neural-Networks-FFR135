# Artificial Neural Network  Homework 1, Nicole Adamah
# Library for Recognizing digits
import numpy as np
def sgn(x):
    return -1 if x < 0 else 1

class Hopfield:
    def __init__(self, patterns, N, zerodiagonal:bool):
        self.N = N
        self.zerodiagonal = zerodiagonal
        self.patterns = patterns
        self.w = np.zeros((N,N))
        self.hebbs_rule()

    def hebbs_rule(self):
        for x in self.patterns:
            self.w += np.matmul(x, x.T)  # Matrix product of two arrays

        if self.zerodiagonal == True:
                np.fill_diagonal(self.w, 0)
        return self.w / self.N

    def asynchronous(self, state, i):
        update_state = np.copy(state)
        update_state[i,:] = sgn(np.matmul(self.w[i,:], state))

        return update_state

    def update_state(self, state):
        S0 = state
        S1 = state
        while True:
            for i in range(self.N):
                S1 = self.asynchronous(S1,i)
            if np.all(S0 == S1):
                return S1
            S0 = S1
