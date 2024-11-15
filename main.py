"""Imports"""

import csv
from random import random
from math import exp
import numpy as np


# Парсинг даты в нужный формат(ЭТО ВСЕ ЕЩЁ НЕ НУЖНЫЙ ФОРМАТ)
def parse_data(name: str = "mnist_test.csv") -> tuple[np.ndarray]:
    """
    Парсинг даты в нужный формат(ЭТО ВСЕ ЕЩЁ НЕ НУЖНЫЙ ФОРМАТ)
    """
    x, y = [], []
    with open(name, "r+", encoding="UTF8") as f:
        for i in csv.reader(f):
            try:
                base = [0] * 10
                base[int(i[0])] = 1
                y.append(base)
                x.append([float(j) for j in i[1:]])
            except ValueError:
                continue
    return np.array(x), np.array(y)


def sigmoid(x):
    """
    Activation function
    """
    return 1 / (1 + exp(-x))


def sigmoid_der(x):
    return x * (1 - x)


def train(x: np.ndarray, y: np.ndarray, layrs: list = [14, 7, 10], ep: int = 10):
    """
    Function for training ns
    """
    weights = []
    layrs = [x.shape[1]] + layrs  # Входной слой
    # Создание весов
    for i in range(len(layrs) - 1):
        weights.append(np.random.rand(layrs[i], layrs[i + 1]))

    print([(len(i), len(i[0])) for i in weights])

    # Проходим по эпохам
    for _ in range(ep):
        for _ in range(len(x)):
            z = []
            sigmas = []
            errors = []

            # Выбираем случайный пример
            i = np.random.randint(0, len(x) - 1)
            a = x[i]

            # Прямой проход через слои нейронной сети
            for j in weights:
                a = a.dot(j)
                z.append(a)
                a = np.array([sigmoid(h) for h in a])
                sigmas.append(a)

            # Начинаем вычислять ошибки и градиенты для обратного распространения
            error = y[i] - sigmas[-1]  # Ошибка для выходного слоя
            errors.append(error)
            print(error.shape)
            print([i.shape for i in sigmas])
            # Вычисляем градиенты для последующих слоев
            for j in range(len(weights) - 1, 0, -1):
                grad = errors[-1].T * sigmoid_der(sigmas[j])
                errors.append(grad * weights[j])
            print([i.shape for i in errors])
    return weights


def main() -> None:
    """
    main function
    """
    x_test, y_test = parse_data()
    x_train, y_train = parse_data("mnist_train.csv")
    x_train /= 256
    train(x_train, y_train)


if __name__ == "__main__":
    main()
