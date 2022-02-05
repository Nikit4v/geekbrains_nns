import numpy as np
from numpy import ndarray
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import simple_framework
from matplotlib import pyplot as plt


class Sigmoid(simple_framework.Function):
    def __call__(self, x):
        x = np.clip(x, a_min=-500, a_max=500)
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self(x) * (1 - self(x))


def to_one_hot(Y):
    n_col = np.max(Y) + 1
    binarized = np.zeros((len(Y), n_col))
    for i in range(len(Y)):
        binarized[i, Y[i]] = 1
    return binarized


digits = load_digits(return_X_y=True)
plt.figure(figsize=(16, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits[0][i, :].reshape([8, 8]))


class Tang(simple_framework.Function):
    def __call__(self, x: ndarray) -> ndarray:
        return 2 * self(2 * x) - 1

    def derivative(self, x: ndarray) -> ndarray:
        return -self(x) ** 2 + 1


if __name__ == '__main__':
    x, y = digits

    scaler = MinMaxScaler()
    y = y.flatten()
    y = to_one_hot(y)
    net = simple_framework.Network(np.array([32, 10]), Sigmoid(), input_length=64)


    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    x_train = scaler.fit_transform(x_train)
    net.train(x_train, y_train, 1000, learning_rate=0.01)
    p = net.eval(x_test)

    accuracy_test = (np.argmax(p, axis=1) == (l_len := np.argmax(y_test, axis=1))).sum() * 100 / len(l_len)
    print(accuracy_test)
