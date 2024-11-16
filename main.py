"""Imports"""
import csv
import numpy as np

# Парсинг данных
def parse_data(name: str = "mnist_test.csv") -> tuple[np.ndarray]:
    """
    Парсинг даты в нужный формат(ЭТО ВСЕ ЕЩЁ НЕ НУЖНЫЙ ФОРМАТ)
    """
    x, y = [], []
    with open(name, "r", encoding="UTF8") as f:
        for i in csv.reader(f):
            try:
                base = [0] * 10
                base[int(i[0])] = 1
                y.append(base)
                x.append([float(j) for j in i[1:]])
            except ValueError:
                continue
    return np.array(x), np.array(y)


def softmax(x):
    """
    Softmax activation function for multi-class classification
    """
    e_x = np.exp(x - np.max(x))  # Для предотвращения переполнения
    return e_x / e_x.sum(axis=1, keepdims=True)


def sigmoid(x):
    """
    Activation function
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    """
    Activation function derivation
    """
    return x * (1 - x)


def train(x: np.ndarray, y: np.ndarray, mu: float = 0.01, layrs: list = [128, 64, 10], ep: int = 10, logs: bool = True):
    """
    Function for training ns
    """
    weights = []
    biases = []
    layrs = [x.shape[1]] + layrs

    # Xavier/Glorot initialization for weights
    for i in range(len(layrs) - 1):
        weights.append(np.random.randn(layrs[i], layrs[i + 1]) * np.sqrt(2 / (layrs[i] + layrs[i + 1])))
        biases.append(np.zeros((1, layrs[i + 1])))

    for e in range(ep):
        true_pr_count = 0
        count = 0
        indices = np.arange(len(x))
        np.random.shuffle(indices)

        for i in indices:  # Обучение на одном примере за раз
            a = x[i:i + 1]
            y_corr = y[i:i + 1]

            sigmas = []
            errors = []

            # Прямой проход
            sigmas.append(a)
            for j, _ in enumerate(weights):
                a = a.dot(weights[j]) + biases[j]
                a = sigmoid(a) if j != len(weights) - 1 else softmax(a)
                sigmas.append(a)

            # Вычисление ошибки
            error = -(y_corr - sigmas[-1])
            errors.append(error)

            # Обратное распространение ошибки
            for j in range(len(weights) - 1, 0, -1):
                grad = np.dot(errors[-1], weights[j].T) * sigmoid_der(sigmas[j])
                errors.append(grad)

            errors = errors[::-1]

            # Обновление весов и биасов
            for j, _ in enumerate(weights):
                weights[j] -= mu * np.dot(sigmas[j].T, errors[j])
                biases[j] -= mu * np.sum(errors[j], axis=0, keepdims=True)

            # Подсчет точности
            true_pr_count += np.argmax(y_corr, axis=1) == np.argmax(sigmas[-1], axis=1)
            count += 1
        if logs:
            print(f"Epoch {e}, accuracy {true_pr_count / count}")

    return weights, biases


def test(x, y, weights, biases):
    """
    Function for testing datasset
    """
    true_pr_count = 0
    count = 0
    for i, _ in enumerate(x):  # Обучение на одном примере за раз
        a = x[i:i + 1]
        y_corr = y[i:i + 1]
        for j, _ in enumerate(weights):
            a = a.dot(weights[j]) + biases[j]
            a = sigmoid(a) if j != len(weights) - 1 else softmax(a)
        true_pr_count += np.argmax(y_corr, axis=1) == np.argmax(a, axis=1)
        count += 1
    return true_pr_count / count


def main() -> None:
    """
    main function
    """
    x_test, y_test = parse_data()
    x_train, y_train = parse_data("mnist_train.csv")
    x_train /= 255  # Нормализация входных данных
    weights, biases = train(x_train, y_train, ep=2)
    print(f"Accuracy at test_data {test(x_test/255, y_test, weights, biases)}")


if __name__ == "__main__":
    main()
