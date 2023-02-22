# Artificial Neural Network  Homework 1, Nicole Adamah
# Recognizing digits with Hopfields network
import numpy as np
from HopfieldDigit import*
import matplotlib.pyplot as plt

x1=np.array([ [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
              [ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
              [ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
              [ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],
              [ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
              [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] ])

x2=np.array([ [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
              [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
              [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
              [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
              [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
              [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1] ])

x3=np.array([ [ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
              [ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],
              [ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],
              [ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],
              [ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],
              [ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1] ])

x4=np.array([ [ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
              [ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
              [ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
              [ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
              [ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, 1, -1],
              [ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1] ])

x5=np.array([ [ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],
              [ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],
              [ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, 1, 1, 1, 1, 1, 1, -1],[ -1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
              [ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
              [ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],
              [ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1] ])

N = len(x1.flatten())
chosen_pattern = 3 #change number for corresponding input pattern to feed

def feed_pattern(chosen_pattern):

    if chosen_pattern == 1:
        p = np.array([[1, 1, -1, -1, -1, -1, -1, -1, 1, 1], [-1, -1, 1, 1, 1, 1, 1, 1, 1, -1],  [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
       [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[-1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
       [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1], [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
       [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[-1, -1, -1, -1, -1, -1, 1, 1, 1, -1],
       [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], [-1, -1, 1, 1, 1, 1, 1, 1, 1, -1],
       [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1]]).flatten()



    if chosen_pattern == 2:
        p = np.array([[1, 1, -1, -1, 1, -1, 1, 1, -1, -1], [1, 1, -1, -1, 1, -1, 1, 1, -1, -1], [1, 1, -1, -1, 1, -1, 1, 1, -1, -1],
                           [1, 1, -1, -1, 1, -1, 1, 1, -1, -1], [1, 1, -1, -1, 1, -1, 1, 1, -1, -1], [1, 1, -1, -1, 1, -1, 1, 1, -1, -1],
                           [1, 1, -1, -1, 1, -1, 1, 1, -1, -1], [1, 1, -1, 1, -1, 1, -1, 1, -1, -1], [1, 1, -1, 1, -1, 1, -1, 1, -1, -1],
                           [1, -1, 1, -1, 1, -1, 1, 1, -1, -1], [1, -1, 1, -1, 1, -1, 1, 1, -1, -1], [1, -1, 1, -1, 1, -1, 1, 1, -1, -1],
                           [1, -1, 1, -1, 1, -1, 1, 1, -1, -1], [1, -1, 1, -1, 1, -1, 1, 1, -1, -1], [1, -1, 1, -1, 1, -1, 1, 1, -1, -1],
                           [1, -1, 1, -1, 1, -1, 1, 1, -1, -1]]).flatten()
    if chosen_pattern == 3:
        p = np.array([[1, 1, 1, -1, -1, -1, -1, 1, 1, 1], [1, 1, 1, -1, -1, -1, -1, 1, 1, 1], [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
                                   [1, 1, 1, -1, -1, -1, -1, 1, 1, 1], [1, 1, 1, -1, -1, -1, -1, 1, 1, 1], [1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
                                   [1, 1, 1, -1, -1, -1, -1, 1, 1, 1], [1, 1, 1, -1, -1, -1, -1, 1, 1, 1], [1, 1, 1, -1, -1, 1, 1, -1, -1, -1],
                                   [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                                   [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],
                                   [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1]]).flatten()

    return np.reshape(p, (-1, 1))

def pattern_reconizer(p, stored):
    for row, column in enumerate(stored):
        if np.all(p == column):
            return row + 1
        if np.all(p == -column):
            return -(row + 1)
    return len(stored) + 1


def typewriter_shape(p):
    p1 = np.reshape(p, (16, 10))
    return p1

def plot(initial_p, converged_p):
    plt.subplot(121)
    plt.title('initial pattern')
    plt.imshow(initial_p, cmap="gray")
    plt.subplot(122)
    plt.title('converged pattern')
    plt.imshow(converged_p, cmap="gray")
    plt.show()

if __name__ == "__main__":
    #store all patterns together
    stored = [np.reshape(x1, (-1, 1)), np.reshape(x2, (-1, 1)), np.reshape(x3, (-1, 1)), np.reshape(x4, (-1, 1)),
              np.reshape(x5, (-1, 1))]
    # initialize the weight matrix
    H = Hopfield(stored, N, zerodiagonal = True)
    # Get the input pattern
    input = feed_pattern(chosen_pattern)
    # update with asynchronous update until converged
    steady = H.update_state(input)
    # change to typewriter scheme
    new_state = typewriter_shape(steady)
    initial_pattern = typewriter_shape(input)

    recognized_digit = pattern_reconizer(steady, stored)
    print(new_state)
    print(f"\nThe pattern converged to the state {recognized_digit}")
    plot(initial_pattern, new_state)
