# Artificial Neural Network  Homework 1, Nicole Adamah
# Calculating the one-step error probability
from Hopfield import *
import numpy as np
from tqdm import tqdm
patterns = [12, 24, 48, 70, 100, 120]
trials = 10**5
p_errors = []
N = 120

for p in patterns:
    error = 0
    for i in tqdm(range(trials)):

        H = Hopfield(p,N, zerodiagonal=False) # Depending on task, change the Zerodiagonal to true or false
        # Pick a neuron and pattern to feed from random integers of N and p
        random_neuron = np.random.randint(N)
        random_pattern = np.random.randint(p)
        picked_pattern = H.patterns[random_pattern]

        # asynchronous update rule
        S1 = H.asynchronous(picked_pattern, random_neuron) #Distorted pattern
        S0 = picked_pattern[random_neuron] # stored pattern

        if (S1 != S0): # If the distorted pattern doesn't converge to the stored, error counts +1
            error+=1

    One_Step_error = error/trials
    p_errors.append(One_Step_error)
    print(f"The One-step error probability is {One_Step_error: .04f} for p = {p} with trials = {trials}")
