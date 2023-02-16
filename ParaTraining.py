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

batch_size = 128

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

Time = 0

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
    __global__ void LSTM_Train(float *activation_output_error, float *next_activation_error, float *oa, 
    float *cell_activation, float *next_cell_error, float *prev_cell_activation, float *fa, float *ga, float *ia, float *prev_cell_error, float *cell_error, float *A1, float *B1, 
    float *A2, float *B2, float *A3, float *B3, float *A4, float *B4, float *C){ 
        int numARows =""" + str(batch_size) + """; 
        int numAColumns =""" + str(hidden_units) + """;
        int numBRows =""" + str(batch_size) + """;
        int numBColumns =""" + str(hidden_units + input_units) + """;
        int numCRows =""" + str(batch_size) + """;
        int numCColumns =""" + str(hidden_units + input_units) + """;
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
        
        if (Row < numARows && Col < numAColumns){
            cell_error[Row*numAColumns + Col] = (((activation_output_error[Row*numAColumns + Col] + 
            next_activation_error[Row*numAColumns + Col]) * oa[Row*numAColumns + Col]) * 
            (1 - (tanhf(cell_activation[Row*numAColumns + Col]) * tanhf(cell_activation[Row*numAColumns + Col])))) + next_cell_error[Row*numAColumns + Col];
            A1[Row*numAColumns + Col] = ((cell_error[Row*numAColumns + Col] * prev_cell_activation[Row*numAColumns + Col]) * 
                                        fa[Row*numAColumns + Col]) * (1 - fa[Row*numAColumns + Col]);
            prev_cell_error[Row*numAColumns + Col] = cell_error[Row*numAColumns + Col] * fa[Row*numAColumns + Col];
            A2[Row*numAColumns + Col] = ((cell_error[Row*numAColumns + Col] * ga[Row*numAColumns + Col]) * ia[Row*numAColumns + Col]) * (1 - ia[Row*numAColumns + Col]);
            A3[Row*numAColumns + Col] = (((activation_output_error[Row*numAColumns + Col] + next_activation_error[Row*numAColumns + Col]) * tanhf(cell_activation[Row*numAColumns + Col])) * oa[Row*numAColumns + Col]) *
                                        (1 - oa[Row*numAColumns + Col]);
            A4[Row*numAColumns + Col] = (cell_error[Row*numAColumns + Col] * ia[Row*numAColumns + Col]) * (1 - (ga[Row*numAColumns + Col] * ga[Row*numAColumns + Col]));
        }
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
    __global__ void lstmDeriv(float *A1, float *B1, float *C1, float *B2, float *C2, float *B3,
                                float *C3, float *B4, float *C4){ 
        int numARows =""" + str(input_units + hidden_units) + """; 
        int numAColumns =""" + str(batch_size) + """;
        int numBRows =""" + str(batch_size) + """;
        int numBColumns =""" + str(hidden_units) + """;
        int numCRows =""" + str(input_units + hidden_units) + """;
        int numCColumns =""" + str(hidden_units) + """;
        float batch =""" + str(batch_size) + """;
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
            C1[Row*numCColumns + Col] = Cvalue1/batch;
            C2[Row*numCColumns + Col] = Cvalue2/batch;
            C3[Row*numCColumns + Col] = Cvalue3/batch;
            C4[Row*numCColumns + Col] = Cvalue4/batch;
        }
    }
    __global__ void update(float *A1, float *A2, float *B1, float *B2, float *C1, float *C2, float *D1, float *D2, 
    float *E1, float *E2, float *F1, float *G1, float *H1, float *J1, float *K1, float *L1, float *M1, float *N1, float *P1, float *Q1){
        int numARows =""" + str(input_units + hidden_units) + """; 
        int numAColumns =""" + str(hidden_units) + """;
        
        int Row = blockDim.y*blockIdx.y + threadIdx.y;
        int Col = blockDim.x*blockIdx.x + threadIdx.x;

        float learning_rate = 0.001;
        float beta1 = 0.90;
        float betalOne = 1 - beta1; 
        float beta2 = 0.99;
        float betalTwo = 1 - 0.99;
        float eps = 0.0000001;
        
        if (Row < numARows && Col < numAColumns){
            A1[Row*numAColumns + Col] = beta1 * A1[Row*numAColumns + Col] + betalOne * A2[Row*numAColumns + Col];
            B1[Row*numAColumns + Col] = beta1 * B1[Row*numAColumns + Col] + betalOne * B2[Row*numAColumns + Col];
            C1[Row*numAColumns + Col] = beta1 * C1[Row*numAColumns + Col] + betalOne * C2[Row*numAColumns + Col];
            D1[Row*numAColumns + Col] = beta1 * D1[Row*numAColumns + Col] + betalOne * D2[Row*numAColumns + Col];
            E1[Row*numAColumns + Col] = beta1 * E1[Row*numAColumns + Col] + betalOne * E2[Row*numAColumns + Col];
            
            F1[Row*numAColumns + Col] = beta2 * F1[Row*numAColumns + Col] + betalTwo * __powf(A2[Row*numAColumns + Col], 2);
            G1[Row*numAColumns + Col] = beta2 * G1[Row*numAColumns + Col] + betalTwo * __powf(B2[Row*numAColumns + Col], 2);
            H1[Row*numAColumns + Col] = beta2 * H1[Row*numAColumns + Col] + betalTwo * __powf(C2[Row*numAColumns + Col], 2);
            J1[Row*numAColumns + Col] = beta2 * J1[Row*numAColumns + Col] + betalTwo * __powf(D2[Row*numAColumns + Col], 2);
            K1[Row*numAColumns + Col] = beta2 * K1[Row*numAColumns + Col] + betalTwo * __powf(E2[Row*numAColumns + Col], 2);
            
            L1[Row*numAColumns + Col] = L1[Row*numAColumns + Col] - learning_rate * (A1[Row*numAColumns + Col] / (sqrtf(F1[Row*numAColumns + Col]) + eps));
            M1[Row*numAColumns + Col] = M1[Row*numAColumns + Col] - learning_rate * (B1[Row*numAColumns + Col] / (sqrtf(G1[Row*numAColumns + Col]) + eps));
            N1[Row*numAColumns + Col] = N1[Row*numAColumns + Col] - learning_rate * (C1[Row*numAColumns + Col] / (sqrtf(H1[Row*numAColumns + Col]) + eps));
            P1[Row*numAColumns + Col] = P1[Row*numAColumns + Col] - learning_rate * (D1[Row*numAColumns + Col] / (sqrtf(J1[Row*numAColumns + Col]) + eps));
            Q1[Row*numAColumns + Col] = Q1[Row*numAColumns + Col] - learning_rate * (E1[Row*numAColumns + Col] / (sqrtf(K1[Row*numAColumns + Col]) + eps));
        }
    }
    """, arch="sm_70")

bsVoc_vocIn = mod.get_function("matmul1")
bsHddn_hddnVoc = mod.get_function("matmul2")
bsHddnPlusIn_hddnPlusInHddn = mod.get_function("LSTM")
Train1 = mod.get_function("LSTM_Train")
derivs = mod.get_function("lstmDeriv")
update = mod.get_function("update")


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
    global Time
    # Parametros con media 0 y desviación estándar de 0.01
    start_time = time.time()
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
    Time = Time + time.time() - start_time

    return parameters


# single layer lstm
def lstm_cell(batch_dataset, prev_activation_matrix, prev_cell_matrix, parameters):
    global Time
    # INPUT --> HIDDEN
    # parametros
    start_time = time.time()
    Tile_size = 32
    fgw = parameters['fgw']
    igw = parameters['igw']
    ogw = parameters['ogw']
    cgw = parameters['cgw']
    Time = Time + time.time() - start_time
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

        fa = np.empty_like(i1).astype(np.float32)
        ia = np.empty_like(i2).astype(np.float32)
        oa = np.empty_like(i3).astype(np.float32)
        ga = np.empty_like(i4).astype(np.float32)

        cell_memory_matrix = np.empty_like(i4).astype(np.float32)
        activation_matrix = np.empty_like(i4).astype(np.float32)

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

        bsHddnPlusIn_hddnPlusInHddn(g_gpu1, h_gpu1, i_gpu1, h_gpu2, i_gpu2, h_gpu3, i_gpu3, h_gpu4, i_gpu4,
                                    prev_cell_matrix_gpu,
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

    start_time = time.time()
    # Guardar valores de activación para el backpropagation
    lstm_activations = dict()
    lstm_activations['fa'] = fa
    lstm_activations['ia'] = ia
    lstm_activations['oa'] = oa
    lstm_activations['ga'] = ga
    Time = Time + time.time() - start_time

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
    global Time
    start_time = time.time()
    embedding_dataset = np.matmul(batch_dataset, embeddings)
    Time = Time + time.time() - start_time
    # print(batch_dataset.shape, "x", embeddings.shape, "=", embedding_dataset.shape)
    return embedding_dataset


# forward propagation
def forward_propagation(batches, parameters, embeddings):
    # tamaño de batch
    global Time

    start_time = time.time()
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
    Time = Time + time.time() - start_time
    # nombres
    for i in range(len(batches) - 1):
        # primeros caracteres del batch
        start_time = time.time()
        batch_dataset = batches[i]
        Time = Time + time.time() - start_time
        # embebidos
        batch_dataset = get_embeddings(batch_dataset, embeddings)

        start_time = time.time()
        embedding_cache['emb' + str(i)] = batch_dataset
        Time = Time + time.time() - start_time
        # lstm
        lstm_activations, ct, at = lstm_cell(batch_dataset, a0, c0, parameters)

        start_time = time.time()
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
        Time = Time + time.time() - start_time
    return embedding_cache, lstm_cache, activation_cache, cell_cache, output_cache


# costo, perplejidad y precisión
def cal_loss_accuracy(batch_labels, output_cache):
    global Time
    start_time = time.time()
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
    Time = Time + time.time() - start_time

    return perplexity, loss, acc


# calcular errores de output
def calculate_output_cell_error(batch_labels, output_cache, parameters):
    global Time

    # alamcenar errores para t pasos
    start_time = time.time()
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
    Time = Time + time.time() - start_time
    return output_error_cache, activation_error_cache


# error de capa lstm
def calculate_single_lstm_cell_error(activation_output_error, next_activation_error, next_cell_error, parameters,
                                     lstm_activation, cell_activation, prev_cell_activation):
    global Time
    start_time = time.time()
    # output gate error
    oa = lstm_activation['oa']
    ia = lstm_activation['ia']
    ga = lstm_activation['ga']
    fa = lstm_activation['fa']
    # get parameters
    fgw = parameters['fgw']
    igw = parameters['igw']
    cgw = parameters['cgw']
    ogw = parameters['ogw']
    Time = Time + time.time() - start_time
    # cell activation error
    ce = np.empty_like(next_cell_error).astype(np.float32)
    ce_gpu = cuda.mem_alloc(ce.nbytes)
    cuda.memcpy_htod(ce_gpu, ce)

    # prev cell error
    pce = np.empty_like(cell_activation).astype(np.float32)
    pce_gpu = cuda.mem_alloc(pce.nbytes)
    cuda.memcpy_htod(pce_gpu, pce)

    # embedding + hidden activation error
    aoe = activation_output_error.astype(np.float32)
    aoe_gpu = cuda.mem_alloc(aoe.nbytes)
    cuda.memcpy_htod(aoe_gpu, aoe)

    nae = next_activation_error.astype(np.float32)
    nae_gpu = cuda.mem_alloc(nae.nbytes)
    cuda.memcpy_htod(nae_gpu, nae)

    oa_gpu = cuda.mem_alloc(oa.astype(np.float32).nbytes)
    cuda.memcpy_htod(oa_gpu, oa)

    ca = cell_activation.astype(np.float32)
    ca_gpu = cuda.mem_alloc(ca.nbytes)
    cuda.memcpy_htod(ca_gpu, ca)

    nce = next_cell_error.astype(np.float32)
    nce_gpu = cuda.mem_alloc(nce.nbytes)
    cuda.memcpy_htod(nce_gpu, nce)

    pca = prev_cell_activation.astype(np.float32)
    pca_gpu = cuda.mem_alloc(pca.nbytes)
    cuda.memcpy_htod(pca_gpu, pca)

    fa_gpu = cuda.mem_alloc(fa.astype(np.float32).nbytes)
    cuda.memcpy_htod(fa_gpu, fa)

    A1 = np.empty_like(cell_activation).astype(np.float32)
    A1_gpu = cuda.mem_alloc(A1.nbytes)
    cuda.memcpy_htod(A1_gpu, A1)
    B1 = fgw.T.astype(np.float32)
    B1_gpu = cuda.mem_alloc(B1.nbytes)
    cuda.memcpy_htod(B1_gpu, B1)

    ga_gpu = cuda.mem_alloc(ga.astype(np.float32).nbytes)
    cuda.memcpy_htod(ga_gpu, ga)

    ia_gpu = cuda.mem_alloc(ia.astype(np.float32).nbytes)
    cuda.memcpy_htod(ia_gpu, ia)

    A2 = np.empty_like(cell_activation).astype(np.float32)
    A2_gpu = cuda.mem_alloc(A2.nbytes)
    cuda.memcpy_htod(A2_gpu, A2)
    B2 = igw.T.astype(np.float32)
    B2_gpu = cuda.mem_alloc(B2.nbytes)
    cuda.memcpy_htod(B2_gpu, B2)

    A3 = np.empty_like(cell_activation).astype(np.float32)
    A3_gpu = cuda.mem_alloc(A3.nbytes)
    cuda.memcpy_htod(A3_gpu, A3)
    B3 = ogw.T.astype(np.float32)
    B3_gpu = cuda.mem_alloc(B3.nbytes)
    cuda.memcpy_htod(B3_gpu, B3)

    A4 = np.empty_like(cell_activation).astype(np.float32)
    A4_gpu = cuda.mem_alloc(A4.nbytes)
    cuda.memcpy_htod(A4_gpu, A4)
    B4 = cgw.T.astype(np.float32)
    B4_gpu = cuda.mem_alloc(B4.nbytes)
    cuda.memcpy_htod(B4_gpu, B4)

    C = np.empty([A1.shape[0], B1.shape[1]]).astype(np.float32)
    C_gpu = cuda.mem_alloc(C.nbytes)
    cuda.memcpy_htod(C_gpu, C)

    Tile_size = 32
    Train1(aoe_gpu, nae_gpu, oa_gpu, ca_gpu, nce_gpu, pca_gpu, fa_gpu, ga_gpu, ia_gpu, pce_gpu, ce_gpu, A1_gpu, B2_gpu,
           A2_gpu, B2_gpu, A3_gpu, B3_gpu, A4_gpu, B4_gpu,
           C_gpu, block=(Tile_size, Tile_size, 1),
           grid=((C.shape[1] // Tile_size) + 1, (C.shape[0] // Tile_size) + 1, 1))

    embed_activation_error = np.empty_like(C).astype(np.float32)
    cuda.memcpy_dtoh(embed_activation_error, C_gpu)
    prev_cell_error = np.empty_like(pce).astype(np.float32)
    cuda.memcpy_dtoh(prev_cell_error, pce_gpu)

    ef = np.empty_like(A1).astype(np.float32)
    cuda.memcpy_dtoh(ef, A1_gpu)
    ei = np.empty_like(A2).astype(np.float32)
    cuda.memcpy_dtoh(ei, A2_gpu)
    eo = np.empty_like(A3).astype(np.float32)
    cuda.memcpy_dtoh(eo, A3_gpu)
    eg = np.empty_like(A4).astype(np.float32)
    cuda.memcpy_dtoh(eg, A4_gpu)

    cell_error_gpu = np.empty_like(ce).astype(np.float32)
    cuda.memcpy_dtoh(cell_error_gpu, ce_gpu)

    # cell_error = np.multiply(np.multiply(activation_output_error + next_activation_error, oa),
    # tanh_derivative(tanh_activation(cell_activation))) + next_cell_error
    # print("ce = cell_error", np.allclose(cell_error, cell_error_gpu, 1e-01, 1e-01))
    # print(np.amax(np.abs(cell_error - ce)))

    # check1 = np.multiply(cell_error, fa)
    # print("pce = prev_cell_error", np.allclose(prev_cell_error, check1, 1e-01, 1e-01))
    # print(np.amax(np.abs(prev_cell_error - check1)))

    # check2 = np.multiply(np.multiply(np.multiply(cell_error, prev_cell_activation), fa), 1 - fa)
    # print("ef = A1", np.allclose(check2, ef, 1e-01, 1e-01))
    # print(np.amax(np.abs(check2 - ef)))

    # check3 = np.multiply(np.multiply(np.multiply(cell_error, ga), ia), 1 - ia)
    # print("ei = A2", np.allclose(check3, ei, 1e-01, 1e-01))
    # print(np.amax(np.abs(check3 - ei)))

    # check4 = np.multiply(np.multiply(np.multiply(activation_output_error + next_activation_error,
    # tanh_activation(cell_activation)), oa), 1 - oa)
    # print("eo = A3", np.allclose(check4, eo, 1e-01, 1e-01))
    # print(np.amax(np.abs(check4 - eo)))

    # check5 = np.multiply(np.multiply(cell_error, ia), tanh_derivative(ga))
    # print("eg = A4", np.allclose(check5, eg, 1e-01, 1e-01))
    # print(np.amax(np.abs(check5 - eg)))

    start_time = time.time()
    input_hidden_units = fgw.shape[0]
    hidden_units = fgw.shape[1]
    input_units = input_hidden_units - hidden_units

    # prev activation error
    embed_activation_error = embed_activation_error.astype(np.float32)
    prev_activation_error = embed_activation_error[:, input_units:]

    # embedding error
    embed_error = embed_activation_error[:, :input_units]

    # almacenar errores
    lstm_error = dict()
    # forget gate error
    lstm_error['ef'] = ef.astype(np.float32)
    lstm_error['ei'] = ei.astype(np.float32)
    lstm_error['eo'] = eo.astype(np.float32)
    lstm_error['eg'] = eg.astype(np.float32)
    Time = Time + time.time() - start_time

    return prev_activation_error, prev_cell_error.astype(np.float32), embed_error, lstm_error


# derivadas de salidas
def calculate_output_cell_derivatives(output_error_cache, activation_cache, parameters):
    # alamacenar sumatoria de derivadas
    global Time
    start_time = time.time()
    dhow = np.zeros(parameters['how'].shape)

    batch_size = activation_cache['a1'].shape[0]

    for i in range(1, len(output_error_cache) + 1):
        # errore en la salida
        output_error = output_error_cache['eo' + str(i)]

        # activacion de entrada
        activation = activation_cache['a' + str(i)]

        # sumatoria de errores
        dhow += np.matmul(activation.T, output_error) / batch_size
    Time = Time + time.time() - start_time
    return dhow


# derivadas de capa lstm
def calculate_single_lstm_cell_derivatives(lstm_error, embedding_matrix, activation_matrix):
    global Time
    # errore en un t
    start_time = time.time()
    ef = lstm_error['ef']
    ei = lstm_error['ei']
    eo = lstm_error['eo']
    eg = lstm_error['eg']

    # activaciones de entrada del paso t
    concat_matrix = np.concatenate((embedding_matrix, activation_matrix), axis=1)

    batch_size = embedding_matrix.shape[0]
    Time = Time + time.time() - start_time

    A1 = concat_matrix.T.astype(np.float32)
    A1_gpu = cuda.mem_alloc(A1.nbytes)
    cuda.memcpy_htod(A1_gpu, A1)

    B1 = ef.astype(np.float32)
    B1_gpu = cuda.mem_alloc(B1.nbytes)
    cuda.memcpy_htod(B1_gpu, B1)
    C1 = np.empty([A1.shape[0], B1.shape[1]]).astype(np.float32)
    C1_gpu = cuda.mem_alloc(C1.nbytes)
    cuda.memcpy_htod(C1_gpu, C1)

    B2 = ei.astype(np.float32)
    B2_gpu = cuda.mem_alloc(B2.nbytes)
    cuda.memcpy_htod(B2_gpu, B2)
    C2 = np.empty([A1.shape[0], B1.shape[1]]).astype(np.float32)
    C2_gpu = cuda.mem_alloc(C2.nbytes)
    cuda.memcpy_htod(C2_gpu, C2)

    B3 = eo.astype(np.float32)
    B3_gpu = cuda.mem_alloc(B3.nbytes)
    cuda.memcpy_htod(B3_gpu, B3)
    C3 = np.empty([A1.shape[0], B1.shape[1]]).astype(np.float32)
    C3_gpu = cuda.mem_alloc(C3.nbytes)
    cuda.memcpy_htod(C3_gpu, C3)

    B4 = eg.astype(np.float32)
    B4_gpu = cuda.mem_alloc(B4.nbytes)
    cuda.memcpy_htod(B4_gpu, B4)
    C4 = np.empty([A1.shape[0], B1.shape[1]]).astype(np.float32)
    C4_gpu = cuda.mem_alloc(C4.nbytes)
    cuda.memcpy_htod(C4_gpu, C4)

    # derivadas de esta paso t
    Tile_size = 32
    derivs(A1_gpu, B1_gpu, C1_gpu, B2_gpu, C2_gpu, B3_gpu, C3_gpu, B4_gpu, C4_gpu, block=(Tile_size, Tile_size, 1),
           grid=((C1.shape[1] // Tile_size) + 1, (C1.shape[0] // Tile_size) + 1, 1))

    dfgw = np.empty_like(C1).astype(np.float32)
    cuda.memcpy_dtoh(dfgw, C1_gpu)

    digw = np.empty_like(C1).astype(np.float32)
    cuda.memcpy_dtoh(digw, C2_gpu)

    dogw = np.empty_like(C1).astype(np.float32)
    cuda.memcpy_dtoh(dogw, C3_gpu)

    dcgw = np.empty_like(C1).astype(np.float32)
    cuda.memcpy_dtoh(dcgw, C4_gpu)

    start_time = time.time()
    # almacenar derivadas
    derivatives = dict()
    derivatives['dfgw'] = dfgw
    derivatives['digw'] = digw
    derivatives['dogw'] = dogw
    derivatives['dcgw'] = dcgw

    Time = Time + time.time() - start_time
    return derivatives


# BACK PROPAGATION
def backward_propagation(batch_labels, embedding_cache, lstm_cache, activation_cache, cell_cache, output_cache,
                         parameters):
    global Time
    # calcular errores de output
    output_error_cache, activation_error_cache = calculate_output_cell_error(batch_labels, output_cache, parameters)

    start_time = time.time()
    # errore de lstm por paso t
    lstm_error_cache = dict()

    # errores de embedding por paso t
    embedding_error_cache = dict()

    # siguiente error de activacion
    # siguiente error de celula
    eat = np.zeros(activation_error_cache['ea1'].shape)
    ect = np.zeros(activation_error_cache['ea1'].shape)
    Time = Time + time.time() - start_time
    # calcular errores de lstm, para todo paso t
    for i in range(len(lstm_cache), 0, -1):
        # calcular errores de lstm para este paso t
        pae, pce, ee, le = calculate_single_lstm_cell_error(activation_error_cache['ea' + str(i)], eat, ect, parameters,
                                                            lstm_cache['lstm' + str(i)], cell_cache['c' + str(i)],
                                                            cell_cache['c' + str(i - 1)])
        start_time = time.time()
        # almacenar en diccionario
        lstm_error_cache['elstm' + str(i)] = le

        # almacenar el dict
        embedding_error_cache['eemb' + str(i - 1)] = ee

        # actualizar suiguiente activacion
        eat = pae
        ect = pce
        Time = Time + time.time() - start_time

    # derivadas de salida
    derivatives = dict()
    derivatives['dhow'] = calculate_output_cell_derivatives(output_error_cache, activation_cache, parameters)

    # calcular derivadas para cada paso t y almacenar en diccionario
    lstm_derivatives = dict()
    for i in range(1, len(lstm_error_cache) + 1):
        lstm_derivatives['dlstm' + str(i)] = calculate_single_lstm_cell_derivatives(lstm_error_cache['elstm' + str(i)],
                                                                                    embedding_cache['emb' + str(i - 1)],
                                                                                    activation_cache['a' + str(i - 1)])
    start_time = time.time()
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
    Time = Time + time.time() - start_time
    return derivatives, embedding_error_cache


# adam optimization
def update_parameters(parameters, derivatives, V, S, t):
    global Time
    start_time = time.time()
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
    Time = Time + time.time() - start_time
    # calcular v parametros
    # vfgw = (beta1 * vfgw + (1 - beta1) * dfgw)
    A1 = vfgw.astype(np.float32)
    A1_gpu = cuda.mem_alloc(A1.nbytes)
    cuda.memcpy_htod(A1_gpu, A1)
    A2 = dfgw.astype(np.float32)
    A2_gpu = cuda.mem_alloc(A2.nbytes)
    cuda.memcpy_htod(A2_gpu, A2)
    # vigw = (beta1 * vigw + (1 - beta1) * digw)
    B1 = vigw.astype(np.float32)
    B1_gpu = cuda.mem_alloc(B1.nbytes)
    cuda.memcpy_htod(B1_gpu, B1)
    B2 = digw.astype(np.float32)
    B2_gpu = cuda.mem_alloc(B2.nbytes)
    cuda.memcpy_htod(B2_gpu, B2)
    # vogw = (beta1 * vogw + (1 - beta1) * dogw)
    C1 = vogw.astype(np.float32)
    C1_gpu = cuda.mem_alloc(C1.nbytes)
    cuda.memcpy_htod(C1_gpu, C1)
    C2 = dogw.astype(np.float32)
    C2_gpu = cuda.mem_alloc(C2.nbytes)
    cuda.memcpy_htod(C2_gpu, C2)
    # vcgw = (beta1 * vcgw + (1 - beta1) * dcgw)
    D1 = vcgw.astype(np.float32)
    D1_gpu = cuda.mem_alloc(D1.nbytes)
    cuda.memcpy_htod(D1_gpu, D1)
    D2 = dcgw.astype(np.float32)
    D2_gpu = cuda.mem_alloc(D2.nbytes)
    cuda.memcpy_htod(D2_gpu, D2)
    # vhow = (beta1 * vhow + (1 - beta1) * dhow)
    E1 = vhow.astype(np.float32)
    E1_gpu = cuda.mem_alloc(E1.nbytes)
    cuda.memcpy_htod(E1_gpu, E1)
    E2 = dhow.astype(np.float32)
    E2_gpu = cuda.mem_alloc(E2.nbytes)
    cuda.memcpy_htod(E2_gpu, E2)

    # calcular s parametros
    # sfgw = (beta2 * sfgw + (1 - beta2) * (dfgw ** 2))
    F1 = sfgw.astype(np.float32)
    F1_gpu = cuda.mem_alloc(F1.nbytes)
    cuda.memcpy_htod(F1_gpu, F1)
    # sigw = (beta2 * sigw + (1 - beta2) * (digw ** 2))
    G1 = sigw.astype(np.float32)
    G1_gpu = cuda.mem_alloc(G1.nbytes)
    cuda.memcpy_htod(G1_gpu, G1)
    # sogw = (beta2 * sogw + (1 - beta2) * (dogw ** 2))
    H1 = sogw.astype(np.float32)
    H1_gpu = cuda.mem_alloc(H1.nbytes)
    cuda.memcpy_htod(H1_gpu, H1)
    # scgw = (beta2 * scgw + (1 - beta2) * (dcgw ** 2))
    J1 = scgw.astype(np.float32)
    J1_gpu = cuda.mem_alloc(J1.nbytes)
    cuda.memcpy_htod(J1_gpu, J1)
    # show = (beta2 * show + (1 - beta2) * (dhow ** 2))
    K1 = show.astype(np.float32)
    K1_gpu = cuda.mem_alloc(K1.nbytes)
    cuda.memcpy_htod(K1_gpu, K1)

    # actualizar parametros
    # fgw = fgw - learning_rate * (vfgw / (np.sqrt(sfgw) + 10e-8))
    L1 = fgw.astype(np.float32)
    L1_gpu = cuda.mem_alloc(L1.nbytes)
    cuda.memcpy_htod(L1_gpu, L1)
    # igw = igw - learning_rate * (vigw / (np.sqrt(sigw) + 10e-8))
    M1 = igw.astype(np.float32)
    M1_gpu = cuda.mem_alloc(M1.nbytes)
    cuda.memcpy_htod(M1_gpu, M1)
    # ogw = ogw - learning_rate * (vogw / (np.sqrt(sogw) + 10e-8))
    N1 = ogw.astype(np.float32)
    N1_gpu = cuda.mem_alloc(N1.nbytes)
    cuda.memcpy_htod(N1_gpu, N1)
    # cgw = cgw - learning_rate * (vcgw / (np.sqrt(scgw) + 10e-8))
    P1 = cgw.astype(np.float32)
    P1_gpu = cuda.mem_alloc(P1.nbytes)
    cuda.memcpy_htod(P1_gpu, P1)
    # how = how - learning_rate * (vhow / (np.sqrt(show) + 10e-8))
    Q1 = how.astype(np.float32)
    Q1_gpu = cuda.mem_alloc(Q1.nbytes)
    cuda.memcpy_htod(Q1_gpu, Q1)

    Tile_size = 32
    update(A1_gpu, A2_gpu, B1_gpu, B2_gpu, C1_gpu, C2_gpu, D1_gpu, D2_gpu, E1_gpu, E2_gpu, F1_gpu, G1_gpu, H1_gpu, J1_gpu, K1_gpu, L1_gpu, M1_gpu, N1_gpu, P1_gpu, Q1_gpu, block=(Tile_size, Tile_size, 1),
           grid=((A1.shape[1] // Tile_size) + 1, (A1.shape[0] // Tile_size) + 1, 1))

    vfgw = np.empty_like(A1).astype(np.float32)
    cuda.memcpy_dtoh(vfgw, A1_gpu)
    vigw = np.empty_like(B1).astype(np.float32)
    cuda.memcpy_dtoh(vigw, B1_gpu)
    vogw = np.empty_like(C1).astype(np.float32)
    cuda.memcpy_dtoh(vogw, C1_gpu)
    vcgw = np.empty_like(D1).astype(np.float32)
    cuda.memcpy_dtoh(vcgw, D1_gpu)
    vhow = np.empty_like(E1).astype(np.float32)
    cuda.memcpy_dtoh(vhow, E1_gpu)

    sfgw = np.empty_like(F1).astype(np.float32)
    cuda.memcpy_dtoh(sfgw, F1_gpu)
    sigw = np.empty_like(G1).astype(np.float32)
    cuda.memcpy_dtoh(sigw, G1_gpu)
    sogw = np.empty_like(H1).astype(np.float32)
    cuda.memcpy_dtoh(sogw, H1_gpu)
    scgw = np.empty_like(J1).astype(np.float32)
    cuda.memcpy_dtoh(scgw, J1_gpu)
    show = np.empty_like(K1).astype(np.float32)
    cuda.memcpy_dtoh(show, K1_gpu)

    fgw = np.empty_like(L1).astype(np.float32)
    cuda.memcpy_dtoh(fgw, L1_gpu)
    igw = np.empty_like(M1).astype(np.float32)
    cuda.memcpy_dtoh(igw, M1_gpu)
    ogw = np.empty_like(N1).astype(np.float32)
    cuda.memcpy_dtoh(ogw, N1_gpu)
    cgw = np.empty_like(P1).astype(np.float32)
    cuda.memcpy_dtoh(cgw, P1_gpu)
    how = np.empty_like(Q1).astype(np.float32)
    cuda.memcpy_dtoh(how, Q1_gpu)

    start_time = time.time()
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
    Time = Time + time.time() - start_time

    return parameters, V, S


# Embeddings
def update_embeddings(embeddings, embedding_error_cache, batch_labels):
    global Time
    start_time = time.time()
    # embeddings derivatives
    embedding_derivatives = np.zeros(embeddings.shape)

    batch_size = batch_labels[0].shape[0]

    # sumatoria de embedding derivatives
    for i in range(len(embedding_error_cache)):
        embedding_derivatives += np.matmul(batch_labels[i].T, embedding_error_cache['eemb' + str(i)]) / batch_size

    # actualizar pesos de embeddings
    embeddings = embeddings - learning_rate * embedding_derivatives
    Time = Time + time.time() - start_time
    return embeddings


def initialize_V(parameters):
    global Time
    start_time = time.time()
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
    Time = Time + time.time() - start_time
    return V


def initialize_S(parameters):
    global Time
    start_time = time.time()
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
    Time = Time + time.time() - start_time
    return S


# train function
def train(train_dataset, iters=1000, batch_size=20):
    global Time
    # parameters
    parameters = initialize_parameters()

    # V y S para Adam
    V = initialize_V(parameters)
    S = initialize_S(parameters)

    start_time = time.time()
    # embeddings
    embeddings = np.random.normal(0, 0.01, (len(vocab), input_units))

    # medidas
    J = []
    P = []
    A = []
    Time = Time + time.time() - start_time
    for step in range(iters):
        start_time = time.time()
        # batch dataset
        index = step % len(train_dataset)
        batches = train_dataset[index]
        Time = Time + time.time() - start_time
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

        start_time = time.time()
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
        Time = Time + time.time() - start_time
    return embeddings, parameters, J, P, A


batch_sizee = batch_size

start_time = time.time()
print("Batch 64, Shared mem tile size 32, 256 hidden")
embeddings, parameters, J, P, A = train(train_dataset, iters=5000, batch_size=batch_sizee)
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
print(Time)
