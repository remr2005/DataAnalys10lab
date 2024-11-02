import csv
import numpy as np

# Парсинг даты в нужный формат(ЭТО ВСЕ ЕЩЁ НЕ НУЖНЫЙ ФОРМАТ)
def parse_data(name:str="mnist_test.csv") -> tuple[np.ndarray]:
    x, y, n = [], [], 0
    with open(name) as f:
        for i in csv.reader(f):
            try:
                y.append(int(i[0]))
                x.append([float(j) for j in i[1:]])
                n+=1
            except ValueError:
                continue
    return (np.array(x), np.array(y))

def main() -> None:
    x_test, y_test = parse_data()
    x_train, y_train = parse_data("mnist_train.csv")
    
if __name__ == "__main__":
    main()
