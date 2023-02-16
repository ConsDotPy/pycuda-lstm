import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule, DynamicSourceModule
import numpy as np
from sys import getsizeof
import time

start = cuda.Event()
end = cuda.Event()


def sigmoid(x):
    return 1 / (1 + np.exp(-x, dtype=np.float32))


def tanh_activation(x):
    return np.tanh(x)


def deviceData():
    (free, total) = cuda.mem_get_info()
    print("Global memory occupancy:%f%% free" % (free * 100 / total))

    for devicenum in range(cuda.Device.count()):
        device = cuda.Device(devicenum)
        attrs = device.get_attributes()

        # Beyond this point is just pretty printing
        print("\n===Attributes for device %d" % devicenum)
        for (key, value) in attrs.items():
            print("%s:%s" % (str(key), str(value)))


Tile_size = 32

m = 256
n = 128
o = 64



start_time = time.time()

g1 = np.random.randn(m, n)
h1 = np.random.randn(n, o)
h2 = np.random.randn(n, o)
h3 = np.random.randn(n, o)
h4 = np.random.randn(n, o)

print("--- %s CPU Create data seconds ---" % (time.time() - start_time))

mod = SourceModule(
    """
    __global__ void gpu_matrix_mult(float *A1, float *B1, float *fa, float *B2, float *ia,float *B3, float *oa, 
                float *B4, float *ga, float *prev_cell_matrix, float *cell_memory_matrix, float *activation_matrix) {
        int m =""" + str(m) + """;
        int n =""" + str(n) + """;
        int k =""" + str(o) + """;
        
        int row = blockIdx.y * blockDim.y + threadIdx.y; 
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        int sum1 = 0;
        int sum2 = 0;
        int sum3 = 0;
        int sum4 = 0;
        
        if( col < k && row < m) 
        {
            for(int i = 0; i < n; i++) 
            {
                sum1 += A1[row * n + i] * B1[i * k + col];
                sum2 += A1[row * n + i] * B2[i * k + col];
                sum3 += A1[row * n + i] * B3[i * k + col];
                sum4 += A1[row * n + i] * B4[i * k + col];
            }
            fa[row * k + col] = __fdividef(1.0, 1 + __expf(-1*sum1));
            ia[row * k + col] = __fdividef(1.0, 1 + __expf(-1*sum2));
            oa[row * k + col] = __fdividef(1.0, 1 + __expf(-1*sum3));
            ga[row * k + col] = tanhf(sum4);
            
            cell_memory_matrix[row * k + col] = fa[row * k + col]*prev_cell_matrix[row * k + col] + ia[row * k + col]*
                                                                                                    ga[row * k + col];
                                        
            activation_matrix[row * k + col] = oa[row * k + col]*tanhf(cell_memory_matrix[row * k + col]);
        }
    }

    __global__ void matrixMultiplyShared(float *A1, float *B1, float *fa, float *B2, float *ia,float *B3, float *oa, 
                float *B4, float *ga, float *prev_cell_matrix, float *cell_memory_matrix, float *activation_matrix){ 
        int numARows =""" + str(g1.shape[0]) + """; 
        int numAColumns =""" + str(g1.shape[1]) + """;
        int numBRows =""" + str(h1.shape[0]) + """;
        int numBColumns =""" + str(h1.shape[1]) + """;
        int numCRows =""" + str(g1.shape[0]) + """;
        int numCColumns =""" + str(h1.shape[1]) + """;
        const int Tile_size = """ + str(Tile_size) + """;
        
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
""")
func3 = mod.get_function("matrixMultiplyShared")

start.record()
g1 = g1.astype(np.float32)
h1 = h1.astype(np.float32)
i1 = np.zeros((g1.shape[0], h2.shape[1]))
i1 = i1.astype(np.float32)
h2 = h2.astype(np.float32)
i2 = np.zeros((g1.shape[0], h2.shape[1]))
i2 = i2.astype(np.float32)
h3 = h3.astype(np.float32)
i3 = np.zeros((g1.shape[0], h3.shape[1]))
i3 = i3.astype(np.float32)
h4 = h4.astype(np.float32)
i4 = np.zeros((g1.shape[0], h4.shape[1]))
i4 = i4.astype(np.float32)

prev_cell_matrix = np.random.randn(m, o)
prev_cell_matrix = prev_cell_matrix.astype(np.float32)

cell_memory_matrix = np.random.randn(m, o)
cell_memory_matrix = cell_memory_matrix.astype(np.float32)

activation_matrix = np.random.randn(m, o)
activation_matrix = activation_matrix.astype(np.float32)

matMul1 = np.empty_like(i1)
matMul2 = np.empty_like(i2)
matMul3 = np.empty_like(i3)
matMul4 = np.empty_like(i4)

cmm = np.empty_like(i4)
activations = np.empty_like(i4)

BLOCK_SIZE = 32
dimGrid = ((o + BLOCK_SIZE - 1) // BLOCK_SIZE, (m + BLOCK_SIZE - 1) // BLOCK_SIZE,1)
dimBlock = (BLOCK_SIZE, BLOCK_SIZE, 1)

for _ in range(5):
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

    #func3(g_gpu1, h_gpu1, i_gpu1, h_gpu2, i_gpu2, h_gpu3, i_gpu3, h_gpu4, i_gpu4, prev_cell_matrix_gpu,
    #cell_memory_matrix_gpu, activation_matrix_gpu, block=(Tile_size, Tile_size, 1), grid=((i4.shape[1] // Tile_size)
    #+ 1, (i4.shape[0] // Tile_size) + 1, 1))
    func3(g_gpu1, h_gpu1, i_gpu1, h_gpu2, i_gpu2, h_gpu3, i_gpu3, h_gpu4, i_gpu4, prev_cell_matrix_gpu,
    cell_memory_matrix_gpu, activation_matrix_gpu, block=dimBlock, grid=dimGrid)
    cuda.memcpy_dtoh(matMul1, i_gpu1)
    cuda.memcpy_dtoh(matMul2, i_gpu2)
    cuda.memcpy_dtoh(matMul3, i_gpu3)
    cuda.memcpy_dtoh(matMul4, i_gpu4)

    cuda.memcpy_dtoh(cmm, cell_memory_matrix_gpu)
    cuda.memcpy_dtoh(activations, activation_matrix_gpu)

end.record()
end.synchronize()

millis = start.time_till(end)
print("--- %s GPU Data Transfer seconds ---" % (millis / 1000))


start_time = time.time()
for _ in range(5):
    C1 = sigmoid(np.matmul(g1, h1))
    C2 = sigmoid(np.matmul(g1, h2))
    C3 = sigmoid(np.matmul(g1, h3))
    C4 = tanh_activation(np.matmul(g1, h4))

    cemema = np.multiply(C1, prev_cell_matrix) + np.multiply(C2, C4)
    activation_matrix = np.multiply(C3, tanh_activation(cemema))
Time = time.time() - start_time
print("--- %s x100 CPU Matmul seconds ---" % (Time))

print(np.allclose(C1, matMul1, 1e-05, 1e-04))
print(np.allclose(C2, matMul2, 1e-05, 1e-04))
print(np.allclose(C3, matMul3, 1e-05, 1e-04))
print(np.allclose(C4, matMul4, 1e-05, 1e-04))

print(np.allclose(cemema, cmm, 1e-05, 1e-04))
print(np.allclose(activation_matrix, activations, 1e-05, 1e-04))
