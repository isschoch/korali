import numpy as np


def BinStep(QPrev):
    N = np.shape(QPrev)[0]
    M = np.shape(QPrev)[1]

    Q = np.zeros(shape=(N // 2, M))
    Q = 0.5 * (QPrev[0:(N-1):2, :] + QPrev[1:N:2, :])

    return Q


def PerformBinning(Q):
    NumLevels = int(np.log2(np.shape(Q)[0]))
    DeltaOrderParam = np.zeros(NumLevels, dtype=float)

    for l in range(NumLevels):
        QAvg = np.average(Q, axis=0)
        M_l = np.shape(Q)[0]

        DeltaOrderParam[l] = np.linalg.norm(
            1. / (M_l * (M_l - 1)) * (Q - QAvg), ord=2, axis=0)

        Q = BinStep(Q)

    return DeltaOrderParam
