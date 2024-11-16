"""Imports"""

import csv
import numpy as np

# Парсинг даты в нужный формат (ЭТО ВСЕ ЕЩЁ НЕ НУЖНЫЙ ФОРМАТ)
def parse_data(name: str = "mnist_test.csv") -> tuple[np.ndarray]:
    """
    Парсинг даты в нужный формат (ЭТО ВСЕ ЕЩЁ НЕ НУЖНЫЙ ФОРМАТ)
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
    return 1 / (1 + np.exp(-x))  # Исправлено использование np.exp вместо exp

def sigmoid_der(x):
    """
    Derivative function
    """
    return x * (1 - x)  # Упрощенная запись, учитывая, что на входе уже a

def train(x: np.ndarray, y: np.ndarray, mu: float = 0.01, layrs: list = [14, 7, 10], ep: int = 10):
    """
    Function for training ns
    """
    weights = []
    layrs = [x.shape[1]] + layrs  # Входной слой
    # Создание весов
    for i in range(len(layrs) - 1):
        weights.append(np.random.rand(layrs[i], layrs[i + 1]))

    # Проходим по эпохам
    for e in range(ep):
        true_pr_count = 0
        count = 0
        for _ in range(len(x)):
            count += 1
            sigmas = []
            errors = []

            # Выбираем случайный пример
            i = np.random.randint(0, len(x) - 1)
            a = x[i].reshape(1, -1)  # Вектор входных данных
            sigmas.append(a)  # Добавляем входной слой в sigmas для удобства

            # Прямой проход через слои нейронной сети
            for j in weights:
                a = a.dot(j)
                a = sigmoid(a)
                sigmas.append(a)

            # Начинаем вычислять ошибки и градиенты для обратного распространения
            if np.argmax(y[i]) == np.argmax(sigmas[-1]):
                true_pr_count += 1
            error = (y[i] - sigmas[-1])  # Ошибка для выходного слоя
            errors.append(sigmoid_der(sigmas[-1]) * error)

            # Вычисляем градиенты для последующих слоев
            for j in range(len(weights) - 1, 0, -1):
                grad = sigmoid_der(sigmas[j]) * np.dot(errors[-1], weights[j].T)
                errors.append(grad)

            errors = errors[::-1]

            # Обновление весов
            for j, _ in enumerate(weights):
                if j == 0:
                    # Обновление весов первого слоя, используя входной вектор x[i]
                    weights[j] -= mu * np.dot(x[i].reshape(-1, 1), errors[j])
                else:
                    weights[j] -= mu * np.dot(sigmas[j].T, errors[j])
        print(f"Epoch {e}, accuracy {true_pr_count/count}" )
    return weights

def main() -> None:
    """
    Main function
    """
    x_test, y_test = parse_data()
    x_train, y_train = parse_data("mnist_train.csv")
    x_train /= 256
    train(x_train, y_train)

if __name__ == "__main__":
    main()
