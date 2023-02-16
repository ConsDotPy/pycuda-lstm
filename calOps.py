import matplotlib.pyplot as plt

ITER = 5000
# Batch
# M = 128
# Hidden
# K = M
# Hidden + InUnits
UNITS = 128
# N = K + InUnits
# VOCAB
VOCAB = 46
# TimeSteps
TIMESTEPS = 22


def lstm_training_ops(M):
    """This module provides a set of functions to compute the number of FLOPs of a LSTM model."""
    K = M
    N = K + UNITS
    result = ITER * (
        M * M * (12 * N * K + 20 * K - 1) + TIMESTEPS * (5 * M + 17 * M * VOCAB + 2 * VOCAB * K * M - VOCAB * K
                                                         + 16 * M * N * K + 33 * M * K + 4 * N * K + 2 * VOCAB * UNITS * K + VOCAB * UNITS)
        + 70 * N * K)
    return result/M


batchHidden = [i for i in range(32, 512, 32)]
ops = [lstm_training_ops(batch) for batch in batchHidden]
# Add some text for labels, title and custom x-axis tick labels, etc.
plt.title('Números de FLOPs por tamaño de LSTM')
plt.plot(batchHidden, ops)
plt.ylabel('FLOP')
plt.xlabel('Batch - Hidden 128 Size')
plt.show()
