import matplotlib.pyplot as plt
import matplotlib.patches as mp
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

class SelfOrganisingMap:
    def __init__(self, input_dimensions, output_shape):
        self.w = np.random.random((*output_shape, input_dimensions))
        self.output_array = output_shape

    def initialize(self, x):
        dist = np.sum((self.w - x)**2, axis=-1)
        return np.unravel_index(np.argmin(dist), self.output_array)

    def neighbourhood_function(self, i, i0, n):
        return np.exp(-np.sum( (np.array(i) - np.array(i0))**2 ) / (2 * n ** 2))

    def train(self, x, learnrate, n):
        i0 = self.initialize(x)
        deltaw = np.empty_like(self.w)
        for i in range(self.output_array[0]):
            for j in range(self.output_array[1]):
                deltaw[i,j,:] = learnrate * self.neighbourhood_function((i, j), i0, n) * (x - self.w[i, j, :])
        self.w += deltaw

def read_data(file_name):
    return pd.read_csv(file_name, delimiter=',', header=None).values

def standardize(data):
    return data / np.max(data)

def train(som, data, epochs, initial_learnrate, learnrate_decay, n, n_decay):
    b_size, _ = data.shape
    for epoch in tqdm(range(epochs)):
        learnrate = initial_learnrate * np.exp(-learnrate_decay * epoch)
        n1 = n * np.exp(-n_decay * epoch)
        for _ in range(b_size):
            x = data[np.random.randint(b_size), :]
            som.train(x, learnrate, n1)
    return som

def plot(neurons, targets, shape, a_plot,iris_flowers, colors):
    img = np.zeros((*shape, 4))
    for (label,), (x, y) in zip(targets, neurons):
        iris = iris_flowers[label]
        img[x, y] = colors[iris]
    a_plot.imshow(img, origin='lower')

def main():
    iris_data = read_data('iris-data.csv')
    iris_labels = read_data('iris-labels.csv')
    iris_data = standardize(iris_data)
    output_array = (40, 40)
    epochs = 10
    input_dimensions = 4
    som = SelfOrganisingMap(input_dimensions, output_array)
    # Using the initial weights
    map1 = np.array([som.initialize(x) for x in iris_data])
    # Train to get weights to make a new map
    initial_learnrate = 0.1
    learnrate_decay = 0.01
    n = 10
    n_decay = 0.05
    train(som, iris_data, epochs, initial_learnrate,learnrate_decay, n, n_decay)
    map2 = np.array([som.initialize(x) for x in iris_data])
    # Making the plots
    iris_flowers = {
        0.0: 'Iris Setosa',
        1.0: 'Iris Versicolour',
        2.0: 'Iris Virginica',}
    colors = {
        'Iris Setosa': (1.0, 0.0, 0.0, 1.0),  # red
        'Iris Versicolour': (0.0, 1.0, 0.0, 1.0),  # green
        'Iris Virginica': (0.0, 0.0, 1.0, 1.0),}  # blue
    fig, (plot1, plot2) = plt.subplots(1, 2)
    plot(map1, iris_labels, output_array, plot1, iris_flowers, colors)
    plot(map2, iris_labels, output_array, plot2, iris_flowers, colors)
    legends = [mp.Patch(color=colors[iris], label=iris) for iris in iris_flowers.values()]
    fig.legend(handles=legends, loc='lower center', bbox_to_anchor=(0.6, 0.9), ncol=len(iris_flowers))
    plt.title('After learning ')
    plt.show()

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))

    main()
