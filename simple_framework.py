import abc
import numpy as np
from typing import *
import tqdm
from numpy import ndarray


class Function(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: ndarray) -> ndarray:
        pass

    @abc.abstractmethod
    def derivative(self, x: ndarray) -> ndarray:
        pass


class Layer:
    """
    Simple layer representation
    """
    activation: Function
    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, num_nodes: int, last_layer_num_nodes: int, activation: Function,
                 init_weights: Optional[np.ndarray] = None):
        """
        Create layer with `num_nodes` neurons using `initial_weights` if presented (random values otherwise)
        """
        self.activation = activation
        if init_weights is None:
            self.weights = np.random.random([last_layer_num_nodes, num_nodes])
        else:
            self.weights = init_weights

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Do forward pass of nn layer
        """
        out: np.ndarray = (np.dot(x, self.weights))
        out = self.activation(out)
        return out

    def backward(self, x: np.ndarray, y: np.ndarray):
        """
        Do backward pass of nn layer
        """
        pass


class Row:
    x: np.ndarray
    y: np.ndarray

    def __init__(self, data):
        self.x = data[0]
        self.y = data[1]


class Network:
    weights: np.ndarray
    layers: list[Layer]

    def __init__(self, scheme: np.ndarray, activation_function: Function, input_length: Optional[int] = None):
        self.layers = list()
        if not input_length:
            last = scheme[0]
        else:
            last = input_length
        for layer in scheme:
            self.layers.append(Layer(layer, last, activation_function))
            last = layer

    def eval(self, x):
        """
        Evaluate model
        """
        return list(self._forward(x))[-1]

    def _forward(self, x):
        yield x
        for layer in self.layers:
            x = layer.forward(x)
            yield x

    @staticmethod
    def _batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def train(self, x: ndarray, y: ndarray, num_epoch: int = 1, learning_rate: float = 1):
        """
        Train model

        :param x: input batch
        :param y: output batch
        :param num_epoch: number of epoch
        :param learning_rate: learning rate
        :return: Nothing
        """

        errors = []
        metrics = []

        bar = tqdm.trange(num_epoch)
        for _ in bar:
            layers_prediction = list(self._forward(x))

            # обратное распространение (back propagation)
            # с использованием градиентного спуска
            output_error = layers_prediction[-1] - y  # производная функции потерь
            last_layer_grad = output_error * self.layers[-1].activation.derivative(layers_prediction[-1])

            errors = [[]]*(len(self.layers)+1)
            grads = [[]]*(len(self.layers)+1)

            errors[-1] = output_error
            grads[-1] = last_layer_grad

            for i in range(2, len(self.layers)+1):
                errors[-i] = grads[-i+1] @ self.layers[-i+1].weights.T
                grads[-i] = errors[-i] * self.layers[-i+1].activation.derivative(layers_prediction[-i])

            # for i in range(1, len(self.layers)+1):
            for i in range(1, len(self.layers) + 1):
                self.layers[-i].weights -= layers_prediction[-i-1].T.dot(grads[-i]) * learning_rate

            # метрики качества
            preds = np.argmax(layers_prediction[-1], axis=1)
            labels = np.argmax(y, axis=1)
            accuracy = (preds == labels).sum() * 100 / len(labels)
            bar.set_description(str(f"{accuracy:.2f}"))

        return errors, metrics

# class Layer
