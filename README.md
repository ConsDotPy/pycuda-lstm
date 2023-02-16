# PyCUDA LSTM implementation
In this repo with the help of PyCUDA we implement a LSTM network. This includes the forward and back propagation algorithms to train it, in this case we progressed through many versions of the algorithm to optimize it with parallel processing. From simple to more complex, we have the following versions doing the single cycle of forward and backward propagation:
- Simple WMMA version, where we used multiple thread to perform the matrix multiplication, the last one was stored on GPU global memory, this is the most basic version.
- Shared memory version, where we used shared memory to store tiles and the last matrix multiplication, this is a little bit faster than the previous one.
- Shared memory with warps version, which is the same as the previous one but we used warps to perform the matrix multiplication.
- Simple TensorCore version, TensorCores are a new feature of the Volta architecture, which is a new type of matrix multiplication, great performance boost.
- Shared memory with warps and TensorCore version, which is the same as the previous one but we used warps to perform the matrix multiplication. Highest throughput so far. We recommend this version.

All in different files, so you can choose the one you want to use. The trained model were put to test on a sentiment analysis task, where we got a 0.89 accuracy on the test set, F1-score of 0.55 and a loss of 0.795, even if there are little variations between models, this due float and half-float point accuracy, there are more benefits from a faster training, which extends to faster inferences.
This accuracy with the same model on a CPU would take hours to train a high input size, and with the fastest GPU implementation it took less than a hour.

This algorithms were tested on a NVIDIA Tesla V100 GPU, with 32GB of memory. The code is written in Python 3.6 and PyCUDA 2018.1.1.
