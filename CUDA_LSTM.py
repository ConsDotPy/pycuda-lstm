import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule, DynamicSourceModule
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
# load GPU Kernels
import pycuda.gpuarray as gpuarray
from sys import getsizeof

IsTraining = True
# datos
path = r'./FrenchNames.csv'
data = pd.read_csv(path)

# tomar la columna de Names
data['Name'] = data['Name']

# tomar n valores
n = 600000
data = np.array(data['Name'][:n]).reshape(-1, 1)

# lower case
data = [x.lower() for x in data[:, 0]]

data = np.array(data).reshape(-1, 1)
np.random.shuffle(data)

print("Forma de csv = {}".format(data.shape))
print()
print("Nombres : ")
print(data[1:20])

# modificar datos
transform_data = np.copy(data)

# encontrar nombre de mayor longitud
max_length = 0
for index in range(len(data)):
    max_length = max(max_length, len(data[index, 0]))

# rellenar con '.' los demás nombres
for index in range(len(data)):
    length = (max_length - len(data[index, 0]))
    string = '.' * length
    transform_data[index, 0] = ''.join([transform_data[index, 0], string])

print("Datos modificados:")
print(transform_data[1:20])

# identificar vocabulario, letras únicas
vocab = list()
for name in transform_data[:, 0]:
    vocab.extend(list(name))

vocab = set(vocab)
vocab_size = len(vocab)

print("Size Vocabulario = {}".format(len(vocab)))
print("Vocabulario      = {}".format(vocab))

# mapear letra a id e id a letra
char_id = dict()
id_char = dict()

for i, char in enumerate(vocab):
    char_id[char] = i
    id_char[i] = char

print('b-{}, 23-{}'.format(char_id['b'], id_char[23]))

# tamaño de batches
train_dataset = []

batch_size = 512

# data modificada a batches
for i in range(len(transform_data) - batch_size + 1):
    start = i * batch_size
    end = start + batch_size

    # batch data
    batch_data = transform_data[start:end]

    if len(batch_data) != batch_size:
        break

    # codificación one hot para cada letra
    char_list = []
    for k in range(len(batch_data[0][0])):
        batch_dataset = np.zeros([batch_size, len(vocab)])
        for j in range(batch_size):
            name = batch_data[j][0]
            char_index = char_id[name[k]]
            batch_dataset[j, char_index] = 1.0

        char_list.append(batch_dataset)

    train_dataset.append(char_list)

# unidades de entrada
input_units = 128

# unidades de hidden layer
hidden_units = batch_size

# unidades de salida
output_units = vocab_size

# learning rate
learning_rate = 0.001

# beta1 adam: coeficiente de decaemiento exponencial
beta1 = 0.90

# beta2 adam: coeficiente de decaemiento exponencial
beta2 = 0.99

# Kernels aceleración GPU
mod = DynamicSourceModule(
    """
    __global__ void matmul1(float *A, float *B, float *C){   
        int numARows =""" + str(batch_size) + """;
        int numAColumns =""" + str(vocab_size) + """;
        int numBRows =""" + str(vocab_size) + """;
        int numBColumns =""" + str(input_units) + """;
        int numCRows =""" + str(batch_size) + """;
        int numCColumns =""" + str(input_units) + """;

        __shared__ float sA[2][2];
        __shared__ float sB[2][2];

        int Row = blockDim.y*blockIdx.y + threadIdx.y;
        int Col = blockDim.x*blockIdx.x + threadIdx.x;

        float Cvalue = 0.0;

        sA[threadIdx.y][threadIdx.x] = 0.0;
        sB[threadIdx.y][threadIdx.x] = 0.0;

        for (int k = 0; k < (((numAColumns)/ 2)); k++){
            if ( (Row < numARows) && (threadIdx.x + (k*2)) < numAColumns){
                sA[threadIdx.y][threadIdx.x] = A[(Row*numAColumns) + threadIdx.x + (k*2)];
            }
            if ( Col < numBColumns && (threadIdx.y + k*2) < numBRows){
                sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k*2)*numBColumns + Col];
            }
            __syncthreads();

            for (int j = 0; j < 2; ++j){
                Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
            }
            __syncthreads();
        }
        if (Row < numCRows && Col < numCColumns){
            C[Row*numCColumns + Col] = Cvalue;
        }
    }
    __global__ void matmul2(float *A, float *B, float *C){   
        int numARows =""" + str(batch_size) + """;
        int numAColumns =""" + str(hidden_units) + """;
        int numBRows =""" + str(hidden_units) + """;
        int numBColumns =""" + str(vocab_size) + """;
        int numCRows =""" + str(batch_size) + """;
        int numCColumns =""" + str(vocab_size) + """;
    
        __shared__ float sA[2][2];
        __shared__ float sB[2][2];
    
        int Row = blockDim.y*blockIdx.y + threadIdx.y;
        int Col = blockDim.x*blockIdx.x + threadIdx.x;
    
        float Cvalue = 0.0;
    
        sA[threadIdx.y][threadIdx.x] = 0.0;
        sB[threadIdx.y][threadIdx.x] = 0.0;
    
        for (int k = 0; k < (((numAColumns)/ 2)); k++){
            if ( (Row < numARows) && (threadIdx.x + (k*2)) < numAColumns){
                sA[threadIdx.y][threadIdx.x] = A[(Row*numAColumns) + threadIdx.x + (k*2)];
            }
            if ( Col < numBColumns && (threadIdx.y + k*2) < numBRows){
                sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k*2)*numBColumns + Col];
            }
            __syncthreads();
    
            for (int j = 0; j < 2; ++j){
                Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
            }
            __syncthreads();
        }
        if (Row < numCRows && Col < numCColumns){
            C[Row*numCColumns + Col] = __expf(Cvalue);
        }
    }
    __global__ void LSTM(float *A1, float *B1, float *fa, float *B2, float *ia,float *B3, float *oa, 
                float *B4, float *ga, float *prev_cell_matrix, float *cell_memory_matrix, float *activation_matrix){ 
        int numARows =""" + str(batch_size) + """; 
        int numAColumns =""" + str(input_units + hidden_units) + """;
        int numBRows =""" + str(input_units + hidden_units) + """;
        int numBColumns =""" + str(hidden_units) + """;
        int numCRows =""" + str(batch_size) + """;
        int numCColumns =""" + str(hidden_units) + """;
        const int Tile_size = 32;
        
        __shared__ float sA1[Tile_size][Tile_size];
        __shared__ float sB1[Tile_size][Tile_size];
        
        __shared__ float sA2[Tile_size][Tile_size];
        __shared__ float sB2[Tile_size][Tile_size];
        
        __shared__ float sA3[Tile_size][Tile_size];
        __shared__ float sB3[Tile_size][Tile_size];
        
        __shared__ float sA4[Tile_size][Tile_size];
        __shared__ float sB4[Tile_size][Tile_size];
        
        int Row = blockDim.y*blockIdx.y + threadIdx.y;
        int Col = blockDim.x*blockIdx.x + threadIdx.x;

        float Cvalue1 = 0.0;
        float Cvalue2 = 0.0;
        float Cvalue3 = 0.0;
        float Cvalue4 = 0.0;
        
        sA1[threadIdx.y][threadIdx.x] = 0.0;
        sB1[threadIdx.y][threadIdx.x] = 0.0;
        
        sA2[threadIdx.y][threadIdx.x] = 0.0;
        sB2[threadIdx.y][threadIdx.x] = 0.0;
        
        sA3[threadIdx.y][threadIdx.x] = 0.0;
        sB3[threadIdx.y][threadIdx.x] = 0.0;
        
        sA4[threadIdx.y][threadIdx.x] = 0.0;
        sB4[threadIdx.y][threadIdx.x] = 0.0;
        
        for (int k = 0; k < (((numAColumns)/ Tile_size)); k++){
            if ( (Row < numARows) && (threadIdx.x + (k*Tile_size)) < numAColumns){
                sA1[threadIdx.y][threadIdx.x] = A1[(Row*numAColumns) + threadIdx.x + (k*Tile_size)];
                sA2[threadIdx.y][threadIdx.x] = A1[(Row*numAColumns) + threadIdx.x + (k*Tile_size)];
                sA3[threadIdx.y][threadIdx.x] = A1[(Row*numAColumns) + threadIdx.x + (k*Tile_size)];
                sA4[threadIdx.y][threadIdx.x] = A1[(Row*numAColumns) + threadIdx.x + (k*Tile_size)];
            }
            if ( Col < numBColumns && (threadIdx.y + k*Tile_size) < numBRows){
                sB1[threadIdx.y][threadIdx.x] = B1[(threadIdx.y + k*Tile_size)*numBColumns + Col];
                sB2[threadIdx.y][threadIdx.x] = B2[(threadIdx.y + k*Tile_size)*numBColumns + Col];
                sB3[threadIdx.y][threadIdx.x] = B3[(threadIdx.y + k*Tile_size)*numBColumns + Col];
                sB4[threadIdx.y][threadIdx.x] = B4[(threadIdx.y + k*Tile_size)*numBColumns + Col];
            }
            __syncthreads();

            for (int j = 0; j < Tile_size; ++j){
                Cvalue1 += sA1[threadIdx.y][j] * sB1[j][threadIdx.x];
                Cvalue2 += sA2[threadIdx.y][j] * sB2[j][threadIdx.x];
                Cvalue3 += sA3[threadIdx.y][j] * sB3[j][threadIdx.x];
                Cvalue4 += sA4[threadIdx.y][j] * sB4[j][threadIdx.x];
            }
            __syncthreads();
        }
        if (Row < numCRows && Col < numCColumns){
            fa[Row*numCColumns + Col] = __fdividef(1.0, 1 + __expf(-1*Cvalue1));
            ia[Row*numCColumns + Col] = __fdividef(1.0, 1 + __expf(-1*Cvalue2));
            oa[Row*numCColumns + Col] = __fdividef(1.0, 1 + __expf(-1*Cvalue3));
            ga[Row*numCColumns + Col] = tanhf(Cvalue4);
            
            cell_memory_matrix[Row*numCColumns + Col] = fa[Row*numCColumns + Col]*prev_cell_matrix
                                        [Row*numCColumns + Col] + ia[Row*numCColumns + Col]*ga[Row*numCColumns + Col];
                                        
            activation_matrix[Row*numCColumns + Col] = oa[Row*numCColumns + Col]
                                                       *tanhf(cell_memory_matrix[Row*numCColumns + Col]);
        }
    }
    __global__ void LSTM_Train(float *A1, float *B1, float *A2, float *B2, float *A3, float *B3, float *A4, float *B4, 
                                float *C){ 
        int numARows =""" + str(batch_size) + """; 
        int numAColumns =""" + str(batch_size) + """;
        int numBRows =""" + str(batch_size) + """;
        int numBColumns =""" + str(batch_size + input_units) + """;
        int numCRows =""" + str(batch_size) + """;
        int numCColumns =""" + str(batch_size + input_units) + """;
        const int Tile_size = 32;
        
        __shared__ float sA1[Tile_size][Tile_size];
        __shared__ float sB1[Tile_size][Tile_size];
        
        __shared__ float sA2[Tile_size][Tile_size];
        __shared__ float sB2[Tile_size][Tile_size];
        
        __shared__ float sA3[Tile_size][Tile_size];
        __shared__ float sB3[Tile_size][Tile_size];
        
        __shared__ float sA4[Tile_size][Tile_size];
        __shared__ float sB4[Tile_size][Tile_size];
        
        int Row = blockDim.y*blockIdx.y + threadIdx.y;
        int Col = blockDim.x*blockIdx.x + threadIdx.x;

        float Cvalue1 = 0.0;
        float Cvalue2 = 0.0;
        float Cvalue3 = 0.0;
        float Cvalue4 = 0.0;
        
        sA1[threadIdx.y][threadIdx.x] = 0.0;
        sB1[threadIdx.y][threadIdx.x] = 0.0;
        
        sA2[threadIdx.y][threadIdx.x] = 0.0;
        sB2[threadIdx.y][threadIdx.x] = 0.0;
        
        sA3[threadIdx.y][threadIdx.x] = 0.0;
        sB3[threadIdx.y][threadIdx.x] = 0.0;
        
        sA4[threadIdx.y][threadIdx.x] = 0.0;
        sB4[threadIdx.y][threadIdx.x] = 0.0;
        
        for (int k = 0; k < (((numAColumns)/ Tile_size)); k++){
            if ( (Row < numARows) && (threadIdx.x + (k*Tile_size)) < numAColumns){
                sA1[threadIdx.y][threadIdx.x] = A1[(Row*numAColumns) + threadIdx.x + (k*Tile_size)];
                sA2[threadIdx.y][threadIdx.x] = A2[(Row*numAColumns) + threadIdx.x + (k*Tile_size)];
                sA3[threadIdx.y][threadIdx.x] = A3[(Row*numAColumns) + threadIdx.x + (k*Tile_size)];
                sA4[threadIdx.y][threadIdx.x] = A4[(Row*numAColumns) + threadIdx.x + (k*Tile_size)];
            }
            if ( Col < numBColumns && (threadIdx.y + k*Tile_size) < numBRows){
                sB1[threadIdx.y][threadIdx.x] = B1[(threadIdx.y + k*Tile_size)*numBColumns + Col];
                sB2[threadIdx.y][threadIdx.x] = B2[(threadIdx.y + k*Tile_size)*numBColumns + Col];
                sB3[threadIdx.y][threadIdx.x] = B3[(threadIdx.y + k*Tile_size)*numBColumns + Col];
                sB4[threadIdx.y][threadIdx.x] = B4[(threadIdx.y + k*Tile_size)*numBColumns + Col];
            }
            __syncthreads();

            for (int j = 0; j < Tile_size; ++j){
                Cvalue1 += sA1[threadIdx.y][j] * sB1[j][threadIdx.x];
                Cvalue2 += sA2[threadIdx.y][j] * sB2[j][threadIdx.x];
                Cvalue3 += sA3[threadIdx.y][j] * sB3[j][threadIdx.x];
                Cvalue4 += sA4[threadIdx.y][j] * sB4[j][threadIdx.x];
            }
            __syncthreads();
        }
        if (Row < numCRows && Col < numCColumns){
            C[Row*numCColumns + Col] = Cvalue1 + Cvalue2 + Cvalue3 + Cvalue4;
        }
    }
    """)

bsVoc_vocIn = mod.get_function("matmul1")
bsHddn_hddnVoc = mod.get_function("matmul2")
bsHddnPlusIn_hddnPlusInHddn = mod.get_function("LSTM")
Train1 = mod.get_function("LSTM_Train")


# Funciones de activación
# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x, np.float64))


# tanh activation
def tanh_activation(x):
    return np.tanh(x)


# softmax activation
def softmax(x):
    exp_x = np.exp(x)
    exp_x_sum = np.sum(exp_x, axis=1).reshape(-1, 1)
    exp_x = exp_x / exp_x_sum
    return exp_x


# derivada de tanh
def tanh_derivative(x):
    return 1 - (x ** 2)


# Parametros
def initialize_parameters():
    # Parametros con media 0 y desviación estándar de 0.01
    mean = 0
    std = 0.01

    # inicializar parametros de LSTM
    forget_gate_weights = np.random.normal(mean, std, (input_units + hidden_units, hidden_units))
    input_gate_weights = np.random.normal(mean, std, (input_units + hidden_units, hidden_units))
    output_gate_weights = np.random.normal(mean, std, (input_units + hidden_units, hidden_units))
    candidate_gate_weights = np.random.normal(mean, std, (input_units + hidden_units, hidden_units))

    # hidden --> output
    hidden_output_weights = np.random.normal(mean, std, (hidden_units, output_units))

    parameters = dict()
    parameters['fgw'] = forget_gate_weights
    parameters['igw'] = input_gate_weights
    parameters['ogw'] = output_gate_weights
    parameters['cgw'] = candidate_gate_weights
    parameters['how'] = hidden_output_weights

    return parameters


# single layer lstm
def lstm_cell(batch_dataset, prev_activation_matrix, prev_cell_matrix, parameters):
    # INPUT --> HIDDEN
    # parametros
    Tile_size = 32
    fgw = parameters['fgw']
    igw = parameters['igw']
    ogw = parameters['ogw']
    cgw = parameters['cgw']
    # Intregrar operaciones con GPU
    # concatenar input con la activación anterior
    concat_dataset = np.concatenate((batch_dataset, prev_activation_matrix), axis=1)
    if IsTraining:
        g1 = concat_dataset.astype(np.float32)
        h1 = fgw.astype(np.float32)
        i1 = np.zeros((g1.shape[0], h1.shape[1]))
        i1 = i1.astype(np.float32)

        h2 = igw.astype(np.float32)
        i2 = np.zeros((g1.shape[0], h2.shape[1]))
        i2 = i2.astype(np.float32)

        h3 = ogw.astype(np.float32)
        i3 = np.zeros((g1.shape[0], h3.shape[1]))
        i3 = i3.astype(np.float32)

        h4 = cgw.astype(np.float32)
        i4 = np.zeros((g1.shape[0], h4.shape[1]))
        i4 = i4.astype(np.float32)

        prev_cell_matrix = prev_cell_matrix.astype(np.float32)

        fa = np.empty_like(i1)
        ia = np.empty_like(i2)
        oa = np.empty_like(i3)
        ga = np.empty_like(i4)

        cell_memory_matrix = np.empty_like(i4)
        activation_matrix = np.empty_like(i4)

        g_gpu1 = cuda.mem_alloc(g1.nbytes)
        cuda.memcpy_htod(g_gpu1, g1)
        h_gpu1 = cuda.mem_alloc(h1.nbytes)
        cuda.memcpy_htod(h_gpu1, h1)
        i_gpu1 = cuda.mem_alloc(i1.nbytes)
        cuda.memcpy_htod(i_gpu1, i1)

        h_gpu2 = cuda.mem_alloc(h2.nbytes)
        cuda.memcpy_htod(h_gpu2, h2)
        i_gpu2 = cuda.mem_alloc(i2.nbytes)
        cuda.memcpy_htod(i_gpu2, i2)

        h_gpu3 = cuda.mem_alloc(h3.nbytes)
        cuda.memcpy_htod(h_gpu3, h3)
        i_gpu3 = cuda.mem_alloc(i3.nbytes)
        cuda.memcpy_htod(i_gpu3, i3)

        h_gpu4 = cuda.mem_alloc(h4.nbytes)
        cuda.memcpy_htod(h_gpu4, h4)
        i_gpu4 = cuda.mem_alloc(i4.nbytes)
        cuda.memcpy_htod(i_gpu4, i4)

        prev_cell_matrix_gpu = cuda.mem_alloc(prev_cell_matrix.nbytes)
        cuda.memcpy_htod(prev_cell_matrix_gpu, prev_cell_matrix)

        cell_memory_matrix_gpu = cuda.mem_alloc(cell_memory_matrix.nbytes)
        cuda.memcpy_htod(cell_memory_matrix_gpu, cell_memory_matrix)

        activation_matrix_gpu = cuda.mem_alloc(activation_matrix.nbytes)
        cuda.memcpy_htod(activation_matrix_gpu, activation_matrix)

        bsHddnPlusIn_hddnPlusInHddn(g_gpu1, h_gpu1, i_gpu1, h_gpu2, i_gpu2, h_gpu3, i_gpu3, h_gpu4, i_gpu4, prev_cell_matrix_gpu,
             cell_memory_matrix_gpu, activation_matrix_gpu, block=(Tile_size, Tile_size, 1),
             grid=((i4.shape[1] // Tile_size) + 1, (i4.shape[0] // Tile_size) + 1, 1))

        cuda.memcpy_dtoh(fa, i_gpu1)
        cuda.memcpy_dtoh(ia, i_gpu2)
        cuda.memcpy_dtoh(oa, i_gpu3)
        cuda.memcpy_dtoh(ga, i_gpu4)

        cuda.memcpy_dtoh(cell_memory_matrix, cell_memory_matrix_gpu)
        cuda.memcpy_dtoh(activation_matrix, activation_matrix_gpu)

    else:
        # forget gate
        fa = np.matmul(concat_dataset, fgw)
        fa = sigmoid(fa)

        # input gate
        ia = np.matmul(concat_dataset, igw)
        ia = sigmoid(ia)

        # output gate
        oa = np.matmul(concat_dataset, ogw)
        oa = sigmoid(oa)

        # candidate gate
        ga = np.matmul(concat_dataset, cgw)
        ga = tanh_activation(ga)
        # actualizar cell memory
        cell_memory_matrix = np.multiply(fa, prev_cell_matrix) + np.multiply(ia, ga)

        # Salida de hidden layer
        activation_matrix = np.multiply(oa, tanh_activation(cell_memory_matrix))

    # Guardar valores de activación para el backpropagation
    lstm_activations = dict()
    lstm_activations['fa'] = fa
    lstm_activations['ia'] = ia
    lstm_activations['oa'] = oa
    lstm_activations['ga'] = ga

    return lstm_activations, cell_memory_matrix, activation_matrix


def output_cell(activation_matrix, parameters):
    # HIDDEN --> OUTPUT
    # parametros
    how = parameters['how']
    output_matrix = np.matmul(activation_matrix, how)
    output_matrix = softmax(output_matrix)
    # (activation_matrix.shape, "x", how.shape, "=", output_matrix.shape)
    return output_matrix


def get_embeddings(batch_dataset, embeddings):
    embedding_dataset = np.matmul(batch_dataset, embeddings)
    # print(batch_dataset.shape, "x", embeddings.shape, "=", embedding_dataset.shape)
    return embedding_dataset


# forward propagation
def forward_propagation(batches, parameters, embeddings):
    # tamaño de batch
    batch_size = batches[0].shape[0]

    # vectores de activación de cada gate para el back prop.
    lstm_cache = dict()
    activation_cache = dict()
    cell_cache = dict()
    output_cache = dict()
    embedding_cache = dict()

    # inicializar activation_matrix(a0) y cell_matrix(c0)
    a0 = np.zeros([batch_size, hidden_units], dtype=np.float32)
    c0 = np.zeros([batch_size, hidden_units], dtype=np.float32)

    # almacenar activaciones de diccionario
    activation_cache['a0'] = a0
    cell_cache['c0'] = c0

    # nombres
    for i in range(len(batches) - 1):
        # primeros caracteres del batch
        batch_dataset = batches[i]

        # embebidos
        batch_dataset = get_embeddings(batch_dataset, embeddings)
        embedding_cache['emb' + str(i)] = batch_dataset

        # lstm
        lstm_activations, ct, at = lstm_cell(batch_dataset, a0, c0, parameters)

        # output
        ot = output_cell(at, parameters)

        # almacenar las t activaciones para backprop
        lstm_cache['lstm' + str(i + 1)] = lstm_activations
        activation_cache['a' + str(i + 1)] = at
        cell_cache['c' + str(i + 1)] = ct
        output_cache['o' + str(i + 1)] = ot

        # actualizar para la siguiente t
        a0 = at
        c0 = ct

    return embedding_cache, lstm_cache, activation_cache, cell_cache, output_cache


# costo, perplejidad y precisión
def cal_loss_accuracy(batch_labels, output_cache):
    loss = 0
    acc = 0
    prob = 1

    # batch size
    batch_size = batch_labels[0].shape[0]

    # evaluar através de los t pasos
    for i in range(1, len(output_cache) + 1):
        # etiquetas y predicciones
        labels = batch_labels[i]
        pred = output_cache['o' + str(i)]

        prob = np.multiply(prob, np.sum(np.multiply(labels, pred), axis=1).reshape(-1, 1))
        loss += np.sum((np.multiply(labels, np.log(pred)) + np.multiply(1 - labels, np.log(1 - pred))), axis=1).reshape(
            -1, 1)
        acc += np.array(np.argmax(labels, 1) == np.argmax(pred, 1), dtype=np.float32).reshape(-1, 1)

    # ccosto, perplejidad y precisión
    perplexity = np.sum((1 / prob) ** (1 / len(output_cache))) / batch_size
    loss = np.sum(loss) * (-1 / batch_size)
    acc = np.sum(acc) / batch_size
    acc = acc / len(output_cache)

    return perplexity, loss, acc


# calcular errores de output
def calculate_output_cell_error(batch_labels, output_cache, parameters):
    # alamcenar errores para t pasos
    output_error_cache = dict()
    activation_error_cache = dict()
    how = parameters['how']

    # evaluar en t pasos
    for i in range(1, len(output_cache) + 1):
        labels = batch_labels[i]
        pred = output_cache['o' + str(i)]

        # output_error para 't' específico
        error_output = pred - labels

        error_activation = np.matmul(error_output, how.T)

        output_error_cache['eo' + str(i)] = error_output
        activation_error_cache['ea' + str(i)] = error_activation

    return output_error_cache, activation_error_cache


# error de capa lstm
def calculate_single_lstm_cell_error(activation_output_error, next_activation_error, next_cell_error, parameters,
                                     lstm_activation, cell_activation, prev_cell_activation):
    activation_error = activation_output_error + next_activation_error

    # output gate error
    oa = lstm_activation['oa']
    # print("activations", oa.shape)
    ia = lstm_activation['ia']
    ga = lstm_activation['ga']
    fa = lstm_activation['fa']
    # get parameters
    fgw = parameters['fgw']
    # print("weights", fgw.shape)
    igw = parameters['igw']
    cgw = parameters['cgw']
    ogw = parameters['ogw']

    eo = np.multiply(activation_error, tanh_activation(cell_activation))
    eo = np.multiply(np.multiply(eo, oa), 1 - oa)

    # cell activation error
    cell_error = np.multiply(activation_error, oa)
    cell_error = np.multiply(cell_error, tanh_derivative(tanh_activation(cell_activation)))

    cell_error += next_cell_error

    # input gate error
    ei = np.multiply(cell_error, ga)
    ei = np.multiply(np.multiply(ei, ia), 1 - ia)

    # gate gate error
    eg = np.multiply(cell_error, ia)
    eg = np.multiply(eg, tanh_derivative(ga))

    # forget gate error
    ef = np.multiply(cell_error, prev_cell_activation)
    ef = np.multiply(np.multiply(ef, fa), 1 - fa)

    # prev cell error
    prev_cell_error = np.multiply(cell_error, fa)

    # embedding + hidden activation error
    embed_activation_error = np.matmul(ef, fgw.T)
    # print("mults", ef.shape, "X", fgw.T.shape)
    embed_activation_error += np.matmul(ei, igw.T)
    embed_activation_error += np.matmul(eo, ogw.T)
    embed_activation_error += np.matmul(eg, cgw.T)

    input_hidden_units = fgw.shape[0]
    hidden_units = fgw.shape[1]
    input_units = input_hidden_units - hidden_units

    # prev activation error
    prev_activation_error = embed_activation_error[:, input_units:]

    # embedding error
    embed_error = embed_activation_error[:, :input_units]

    # almacenar errores
    lstm_error = dict()
    lstm_error['ef'] = ef
    lstm_error['ei'] = ei
    lstm_error['eo'] = eo
    lstm_error['eg'] = eg

    return prev_activation_error, prev_cell_error, embed_error, lstm_error


# derivadas de salidas
def calculate_output_cell_derivatives(output_error_cache, activation_cache, parameters):
    # alamacenar sumatoria de derivadas
    dhow = np.zeros(parameters['how'].shape)

    batch_size = activation_cache['a1'].shape[0]

    for i in range(1, len(output_error_cache) + 1):
        # errore en la salida
        output_error = output_error_cache['eo' + str(i)]

        # activacion de entrada
        activation = activation_cache['a' + str(i)]

        # sumatoria de errores
        dhow += np.matmul(activation.T, output_error) / batch_size

    return dhow


# derivadas de capa lstm
def calculate_single_lstm_cell_derivatives(lstm_error, embedding_matrix, activation_matrix):
    # errore en un t
    ef = lstm_error['ef']
    ei = lstm_error['ei']
    eo = lstm_error['eo']
    eg = lstm_error['eg']

    # activaciones de entrada del paso t
    concat_matrix = np.concatenate((embedding_matrix, activation_matrix), axis=1)

    batch_size = embedding_matrix.shape[0]

    # derivadas de esta paso t
    dfgw = np.matmul(concat_matrix.T, ef) / batch_size
    digw = np.matmul(concat_matrix.T, ei) / batch_size
    dogw = np.matmul(concat_matrix.T, eo) / batch_size
    dcgw = np.matmul(concat_matrix.T, eg) / batch_size

    # almacenar derivadas
    derivatives = dict()
    derivatives['dfgw'] = dfgw
    derivatives['digw'] = digw
    derivatives['dogw'] = dogw
    derivatives['dcgw'] = dcgw

    return derivatives


# BACK PROPAGATION
def backward_propagation(batch_labels, embedding_cache, lstm_cache, activation_cache, cell_cache, output_cache,
                         parameters):
    # calcular errores de output
    output_error_cache, activation_error_cache = calculate_output_cell_error(batch_labels, output_cache, parameters)

    # errore de lstm por paso t
    lstm_error_cache = dict()

    # errores de embedding por paso t
    embedding_error_cache = dict()

    # siguiente error de activacion
    # siguiente error de celula
    eat = np.zeros(activation_error_cache['ea1'].shape)
    ect = np.zeros(activation_error_cache['ea1'].shape)

    # calcular errores de lstm, para todo paso t
    for i in range(len(lstm_cache), 0, -1):
        # calcular errores de lstm para este paso t
        pae, pce, ee, le = calculate_single_lstm_cell_error(activation_error_cache['ea' + str(i)], eat, ect, parameters,
                                                            lstm_cache['lstm' + str(i)], cell_cache['c' + str(i)],
                                                            cell_cache['c' + str(i - 1)])

        # almacenar en diccionario
        lstm_error_cache['elstm' + str(i)] = le

        # almacenar el dict
        embedding_error_cache['eemb' + str(i - 1)] = ee

        # actualizar suiguiente activacion
        eat = pae
        ect = pce

    # derivadas de salida
    derivatives = dict()
    derivatives['dhow'] = calculate_output_cell_derivatives(output_error_cache, activation_cache, parameters)

    # calcular derivadas para cada paso t y almacenar en diccionario
    lstm_derivatives = dict()
    for i in range(1, len(lstm_error_cache) + 1):
        lstm_derivatives['dlstm' + str(i)] = calculate_single_lstm_cell_derivatives(lstm_error_cache['elstm' + str(i)],
                                                                                    embedding_cache['emb' + str(i - 1)],
                                                                                    activation_cache['a' + str(i - 1)])

    # inicializar derivadas en zeros
    derivatives['dfgw'] = np.zeros(parameters['fgw'].shape)
    derivatives['digw'] = np.zeros(parameters['igw'].shape)
    derivatives['dogw'] = np.zeros(parameters['ogw'].shape)
    derivatives['dcgw'] = np.zeros(parameters['cgw'].shape)

    # sumatoria de derivadas para cada paso
    for i in range(1, len(lstm_error_cache) + 1):
        derivatives['dfgw'] += lstm_derivatives['dlstm' + str(i)]['dfgw']
        derivatives['digw'] += lstm_derivatives['dlstm' + str(i)]['digw']
        derivatives['dogw'] += lstm_derivatives['dlstm' + str(i)]['dogw']
        derivatives['dcgw'] += lstm_derivatives['dlstm' + str(i)]['dcgw']

    return derivatives, embedding_error_cache


# adam optimization
def update_parameters(parameters, derivatives, V, S, t):
    # derivatives
    dfgw = derivatives['dfgw']
    digw = derivatives['digw']
    dogw = derivatives['dogw']
    dcgw = derivatives['dcgw']
    dhow = derivatives['dhow']

    # parameters
    fgw = parameters['fgw']
    igw = parameters['igw']
    ogw = parameters['ogw']
    cgw = parameters['cgw']
    how = parameters['how']

    # V parameters
    vfgw = V['vfgw']
    vigw = V['vigw']
    vogw = V['vogw']
    vcgw = V['vcgw']
    vhow = V['vhow']

    # S parameters
    sfgw = S['sfgw']
    sigw = S['sigw']
    sogw = S['sogw']
    scgw = S['scgw']
    show = S['show']

    # calcular v parametros
    vfgw = (beta1 * vfgw + (1 - beta1) * dfgw)
    vigw = (beta1 * vigw + (1 - beta1) * digw)
    vogw = (beta1 * vogw + (1 - beta1) * dogw)
    vcgw = (beta1 * vcgw + (1 - beta1) * dcgw)
    vhow = (beta1 * vhow + (1 - beta1) * dhow)

    # calcular s parametros
    sfgw = (beta2 * sfgw + (1 - beta2) * (dfgw ** 2))
    sigw = (beta2 * sigw + (1 - beta2) * (digw ** 2))
    sogw = (beta2 * sogw + (1 - beta2) * (dogw ** 2))
    scgw = (beta2 * scgw + (1 - beta2) * (dcgw ** 2))
    show = (beta2 * show + (1 - beta2) * (dhow ** 2))

    # actualizar parametros
    fgw = fgw - learning_rate * (vfgw / (np.sqrt(sfgw) + 10e-8))
    igw = igw - learning_rate * (vigw / (np.sqrt(sigw) + 10e-8))
    ogw = ogw - learning_rate * (vogw / (np.sqrt(sogw) + 10e-8))
    cgw = cgw - learning_rate * (vcgw / (np.sqrt(scgw) + 10e-8))
    how = how - learning_rate * (vhow / (np.sqrt(show) + 10e-8))

    # almacenar nuevos pesos
    parameters['fgw'] = fgw
    parameters['igw'] = igw
    parameters['ogw'] = ogw
    parameters['cgw'] = cgw
    parameters['how'] = how

    # new V parameters
    V['vfgw'] = vfgw
    V['vigw'] = vigw
    V['vogw'] = vogw
    V['vcgw'] = vcgw
    V['vhow'] = vhow

    # new s parameters
    S['sfgw'] = sfgw
    S['sigw'] = sigw
    S['sogw'] = sogw
    S['scgw'] = scgw
    S['show'] = show

    return parameters, V, S


# Embeddings
def update_embeddings(embeddings, embedding_error_cache, batch_labels):
    # embeddings derivatives
    embedding_derivatives = np.zeros(embeddings.shape)

    batch_size = batch_labels[0].shape[0]

    # sumatoria de embedding derivatives
    for i in range(len(embedding_error_cache)):
        embedding_derivatives += np.matmul(batch_labels[i].T, embedding_error_cache['eemb' + str(i)]) / batch_size

    # actualizar pesos de embeddings
    embeddings = embeddings - learning_rate * embedding_derivatives
    return embeddings


def initialize_V(parameters):
    Vfgw = np.zeros(parameters['fgw'].shape)
    Vigw = np.zeros(parameters['igw'].shape)
    Vogw = np.zeros(parameters['ogw'].shape)
    Vcgw = np.zeros(parameters['cgw'].shape)
    Vhow = np.zeros(parameters['how'].shape)

    V = dict()
    V['vfgw'] = Vfgw
    V['vigw'] = Vigw
    V['vogw'] = Vogw
    V['vcgw'] = Vcgw
    V['vhow'] = Vhow
    return V


def initialize_S(parameters):
    Sfgw = np.zeros(parameters['fgw'].shape)
    Sigw = np.zeros(parameters['igw'].shape)
    Sogw = np.zeros(parameters['ogw'].shape)
    Scgw = np.zeros(parameters['cgw'].shape)
    Show = np.zeros(parameters['how'].shape)

    S = dict()
    S['sfgw'] = Sfgw
    S['sigw'] = Sigw
    S['sogw'] = Sogw
    S['scgw'] = Scgw
    S['show'] = Show
    return S


# train function
def train(train_dataset, iters=1000, batch_size=20):
    # parameters
    parameters = initialize_parameters()

    # V y S para Adam
    V = initialize_V(parameters)
    S = initialize_S(parameters)

    # embeddings
    embeddings = np.random.normal(0, 0.01, (len(vocab), input_units))

    # medidas
    J = []
    P = []
    A = []

    for step in range(iters):
        # batch dataset
        index = step % len(train_dataset)
        batches = train_dataset[index]

        # forward propagation
        embedding_cache, lstm_cache, activation_cache, cell_cache, output_cache = forward_propagation(batches,
                                                                                                      parameters,
                                                                                                      embeddings)

        # medidas
        perplexity, loss, acc = cal_loss_accuracy(batches, output_cache)

        # backward propagation
        derivatives, embedding_error_cache = backward_propagation(batches, embedding_cache, lstm_cache,
                                                                  activation_cache, cell_cache, output_cache,
                                                                  parameters)

        # actualizar parametros
        parameters, V, S = update_parameters(parameters, derivatives, V, S, step)

        # actualizar embeddings
        embeddings = update_embeddings(embeddings, embedding_error_cache, batches)

        J.append(loss)
        P.append(perplexity)
        A.append(acc)

        # loss, accuracy and perplexity
        if step % 10 == 0:
            print("Single Batch :")
            print('Paso      = {}'.format(step))
            print('Loss       = {}'.format(round(loss, 2)))
            print('Perplexity = {}'.format(round(perplexity, 2)))
            print('Accuracy   = {}'.format(round(acc * 100, 2)))
            print("--- %s seconds ---" % (time.time() - start_time))
            print()

    return embeddings, parameters, J, P, A


batch_sizee = batch_size

start_time = time.time()
print("Batch 64, Shared mem tile size 32, 256 hidden")
embeddings, parameters, J, P, A = train(train_dataset, iters=2500, batch_size=batch_sizee)
print("--- %s seconds ---" % (time.time() - start_time))

avg_loss = list()
avg_acc = list()
avg_perp = list()
i = 0
while i < len(J):
    avg_loss.append(np.mean(J[i:i + 30]))
    avg_acc.append(np.mean(A[i:i + 30]))
    avg_perp.append(np.mean(P[i:i + 30]))
    i += 30

plt.plot(list(range(len(avg_loss))), avg_loss)
plt.xlabel("x")
plt.ylabel("Loss (Promedio en 30 batches)")
plt.title("Loss")
plt.show()

plt.plot(list(range(len(avg_perp))), avg_perp)
plt.xlabel("x")
plt.ylabel("Perplexity (Promedio en 30 batches)")
plt.title("Perplexity")
plt.show()

plt.plot(list(range(len(avg_acc))), avg_acc)
plt.xlabel("x")
plt.ylabel("Accuracy (Promedio en)")
plt.title("Accuracy")
plt.show()


# predict
def predict(parameters, embeddings, id_char, vocab_size):
    names = []

    # predict 20 names
    for i in range(20):
        # iniciar activation_matrix(a0) y cell_matrix(c0)
        a0 = np.zeros([1, hidden_units], dtype=np.float32)
        c0 = np.zeros([1, hidden_units], dtype=np.float32)

        # blank name
        name = ''

        # batch dataset of single char
        batch_dataset = np.zeros([1, vocab_size])

        index = np.random.randint(0, vocab_size, 1)[0]

        batch_dataset[0, index] = 1.0
        name += id_char[index]

        char = id_char[index]

        # predecir caracteres hasta obtener '.'
        while char != '.' and len(name) < max_length:
            # get embeddings
            batch_dataset = get_embeddings(batch_dataset, embeddings)
            # lstm cell
            lstm_activations, ct, at = lstm_cell(batch_dataset, a0, c0, parameters)

            # output cell
            ot = output_cell(at, parameters)
            pred = np.argmax(ot)
            # lista de nombres predecidos
            name += id_char[pred]

            char = id_char[pred]

            batch_dataset = np.zeros([1, vocab_size])
            batch_dataset[0, pred] = 1.0

            a0 = at
            c0 = ct

        names.append(name)

    return names


IsTraining = False
predict(parameters, embeddings, id_char, vocab_size)
predict(parameters, embeddings, id_char, vocab_size)
predict(parameters, embeddings, id_char, vocab_size)
