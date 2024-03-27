import math


def metric(kpi, hours):
    tmp = 24
    ratio = 0.6

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    print(kpi * ratio + (1 - sigmoid(hours / tmp)) * (1 - ratio))


if __name__ == '__main__':
    kpi = 0.12
    hours = 12

    metric(kpi, hours)