import random
import numpy as np
import matplotlib.pyplot as plt


class Kohonen(object):
    def __init__(self, h, w, alpha_start=0.6, seed=9, r=3):
        """
        Initialize the Kohonen object with a given map size
        :param seed: {int} random seed to use
        :param h: {int} height of the map
        :param w: {int} width of the map
        :param dim: {int} dimensions of the map
        :param alpha_start: {float} initial alpha (learning rate) at training start
        :param sigma_start: {float} initial sigma (restraint / neighborhood function) at training start; if `None`: w / 2
        """
        np.random.seed(seed)
        self.shape = (h, w)
        self.alpha_start = alpha_start
        self.radius_strength = r
        self.d = None

    def fit(self, data, interval=1000, print_mode=5):
        """
        Train the SOM on the given data for several iterations
        :param print_mode:
        :param data: {numpy.ndarray} data to train on
        :param interval: {int} interval of epochs to use for saving training errors
        """
        self.d = data
        self.iteration_limit = interval
        x_min = np.min(data[:, 0])
        y_min = np.min(data[:, 1])
        x_max = np.max(data[:, 0])
        y_max = np.max(data[:, 1])

        self.map = np.array([[(random.uniform(x_min + 0.001, x_max),
                               random.uniform(y_min + 0.001, y_max)) for i in range(self.shape[1])] for j in
                             range(self.shape[0])])
        for t in range(self.iteration_limit):
            # randomly pick an input vector
            n = random.randint(0, len(data) - 1)
            # the input vector who chosen
            bmu = self.best_neuron(data[n])
            if t % print_mode == 0:
                self.draw_map(x_max, x_min, t)
            self.update_map(bmu, data[n], t)

    def best_neuron(self, vector):
        """
        Compute the winner neuron closest to the vector (Euclidean distance)
        :param vector: {numpy.ndarray} vector of current data point(s)
        :return: indices of winning neuron
        """
        min_neuron_dest = np.inf
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                euclid_dist = np.linalg.norm(vector - self.map[x, y])
                if euclid_dist < min_neuron_dest:
                    min_neuron_dest = euclid_dist
                    ans = (x, y)
        return ans

    def update_map(self, bmu, X_i, t):
        """
        Update map by found BMU at iteration t.
        :param bmu: best neuron indices
        :param X_i: sample - target input vector
        :param t: current iteration
        """
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                dist_from_bmu = np.linalg.norm(np.array(bmu) - np.array([x, y]))
                alpha = self.alpha_start * np.exp(-t / 300)  # update alpha
                radius = np.exp(-np.power(dist_from_bmu, 2) / self.radius_strength)  # update radius
                self.map[(x, y)] += alpha * radius * (X_i - self.map[(x, y)])

    def draw_map(self, max, min, t):
        xs = []
        ys = []
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                xs.append(self.map[i, j, 0])
                ys.append(self.map[i, j, 1])

        fig, ax = plt.subplots()
        ax.scatter([xs], [ys], c='r')
        ax.set_xlim(min, max)
        ax.set_ylim(min, max)
        ax.plot(xs, ys, 'b-')
        ax.scatter(self.d[:, 0], self.d[:, 1], alpha=0.3)
        ax.set_title("Data size:" + str(len(self.d)) + " | "
                     + "Iter:" + str(self.iteration_limit) + " | "+
                     "Epoch:" + str(t) +
                     "\n" +
                     "LR:" + str(self.alpha_start) + " | "
                     + "Net size:" + str(self.shape) + " | "
                     + "R:" + str(self.radius_strength))
        plt.show()


def create_data(d_size=1000, condition=1):
    data = np.empty((d_size, 2), dtype=object)
    random.seed(11)
    if condition == 1:
        for i in range(d_size):
            data[i, 0] = random.randint(0, 1000) / 1000
            data[i, 1] = random.randint(0, 1000) / 1000
    elif condition == 2:
        for i in range(d_size):
            flag = random.randint(0, 100)
            if flag < 80:
                data[i, 0] = i / 1000
                data[i, 1] = random.randint(0, i) / 1000
            else:
                data[i, 0] = i / 1000
                data[i, 1] = random.randint(i, 1000) / 1000
    elif condition == 3:
        c = 0
        for i in range(int(d_size * 0.2)):
            data[i, 0] = random.randint(500, 1000) / 1000
            data[i, 1] = random.randint(500, 1000) / 1000
            c = i
        for j in range(c, c + int(d_size * 0.8) + 1):
            data[j, 0] = random.randint(0, 500) / 1000
            data[j, 1] = random.randint(0, 500) / 1000
    else:
        n = 0
        while n < d_size:
            x = random.uniform(-2, 2)
            y = random.uniform(-2, 2)
            if 1 <= x ** 2 + y ** 2 <= 2:
                data[n, 0] = x
                data[n, 1] = y
                n += 1

    return data.astype(np.float64)

def main():
    data1 = create_data(condition=1)
    # Q1.1
    Kohonen(h=1, w=15, r=2, alpha_start=0.35).fit(data=data1, interval=1001, print_mode=500)
    # Q1.2
    Kohonen(h=1, w=200, r=60, alpha_start=0.4).fit(data=data1, interval=1001, print_mode=500)
    # Q1.3.1
    data2 = create_data(condition=2, d_size=1000)
    Kohonen(h=1, w=30, r=50, alpha_start=0.5).fit(data=data2, interval=1001, print_mode=100)
    # Q1.3.2
    data3 = create_data(condition=3, d_size=1000)
    Kohonen(h=1, w=30, r=50, alpha_start=0.5).fit(data=data3, interval=1001, print_mode=100)
    # Q2
    data4 = create_data(condition=4, d_size=1000)
    Kohonen(h=1, w=30, r=15, alpha_start=0.6).fit(data=data4, interval=1001, print_mode=100)


if __name__ == '__main__':
    main()





