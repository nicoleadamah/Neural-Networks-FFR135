import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
class Reservoir:

    def __init__(self, input_neurons, reservoir_neurons):
        self.inputs = input_neurons
        self.outputs = input_neurons
        self.reservoir_size = reservoir_neurons
        self.calc_weights(reservoir_neurons, input_neurons, input_neurons)

    def calc_weights(self,reservoir_size,outputs,inputs):
        self.weights_in = np.random.normal(0, np.sqrt(0.002), size=(reservoir_size, inputs))
        self.weights_r = np.random.normal(0, np.sqrt(2/reservoir_size), size=(reservoir_size, reservoir_size))
        self.weights_out = np.empty((outputs, reservoir_size))

    def ridge_regression(self, x, y, k):
        T = x.shape[1]
        r = np.zeros((self.reservoir_size, T + 1))
        for t in range(0, T):
            r[:, t + 1] = np.tanh(self.weights_r @ r[:, t] + self.weights_in @ x[:, t])
        # Remove initial to avoid fluctuations
        r = r[:, 26:]
        y = y[:, 25:]
        # Using rigid_regression to get the output weights
        self.weights_out = y @ r.T @ np.linalg.inv(r @ r.T + k * np.identity(self.reservoir_size))

    def predict(self, x, steps):
        r = np.zeros((self.reservoir_size,))
        for i in range(x.shape[1]):
            r = np.tanh(self.weights_r @ r + self.weights_in @ x[:, i])
        reservoir_output = np.empty((self.outputs, steps))
        for j in range(steps):
            reservoir_output[:,j] = self.weights_out @ r
            r = np.tanh(self.weights_r @ r + self.weights_in @ reservoir_output[:, j])
        return reservoir_output

def read_data(file_name: str) -> np.ndarray:
    return pd.read_csv(file_name, delimiter=',', header=None).values

def main():
    reservoir_computer = Reservoir(3, 500)
    # training the data with ridge regression
    training_set = read_data('training-set.csv')
    x = training_set[:, :-1]
    y = training_set[:, 1:]
    k = 0.01
    reservoir_computer.ridge_regression(x, y, k)
    #  feeding the test data through the network
    test_set = read_data('test-set-2.csv')
    steps = 500
    test_pred = reservoir_computer.predict(test_set, steps)
    # plot training set and predicted test set
    fig = plt.figure()
    plot_data = plt.axes(projection='3d')
    x, y, z = test_set
    plot_data.plot3D(x, y, z, c='b',label='Actual test-set')
    x2, y2, z2 = test_pred
    plot_data.plot3D(x2, y2, z2, c='g', label='Predicted test-set')
    plt.legend()
    plt.show()
    np.savetxt('prediction.csv', test_pred[1, :], delimiter=',')

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    main()