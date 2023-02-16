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

# softmax activation
def softmax(x):
    exp_x = x
    exp_x_sum = np.sum(exp_x, axis=1).reshape(-1, 1)
    exp_x = exp_x / exp_x_sum
    return exp_x

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

m = 512
n = 512
o = 46


start_time = time.time()

g1 = np.random.randn(m, n)
h1 = np.random.randn(n, o)
h2 = np.random.randn(n, o)
h3 = np.random.randn(n, o)
h4 = np.random.randn(n, o)

print("--- %s CPU Create data seconds ---" % (time.time() - start_time))

mod = SourceModule(
    """
    __global__ void matrixMultiplyShared(float *A1, float *B1, float *fa){ 
        int numARows =""" + str(g1.shape[0]) + """; 
        int numAColumns =""" + str(g1.shape[1]) + """;
        int numBRows =""" + str(h1.shape[0]) + """;
        int numBColumns =""" + str(h1.shape[1]) + """;
        int numCRows =""" + str(g1.shape[0]) + """;
        int numCColumns =""" + str(h1.shape[1]) + """;
        const int Tile_size = """ + str(Tile_size) + """;

        __shared__ float sA1[Tile_size][Tile_size];
        __shared__ float sB1[Tile_size][Tile_size];

        int Row = blockDim.y*blockIdx.y + threadIdx.y;
        int Col = blockDim.x*blockIdx.x + threadIdx.x;

        float Cvalue1 = 0.0;

        sA1[threadIdx.y][threadIdx.x] = 0.0;
        sB1[threadIdx.y][threadIdx.x] = 0.0;

        for (int k = 0; k < (((numAColumns)/ Tile_size)); k++){
            if ( (Row < numARows) && (threadIdx.x + (k*Tile_size)) < numAColumns){
                sA1[threadIdx.y][threadIdx.x] = A1[(Row*numAColumns) + threadIdx.x + (k*Tile_size)];
            }
            if ( Col < numBColumns && (threadIdx.y + k*Tile_size) < numBRows){
                sB1[threadIdx.y][threadIdx.x] = B1[(threadIdx.y + k*Tile_size)*numBColumns + Col];
            }
            __syncthreads();

            for (int j = 0; j < Tile_size; ++j){
                Cvalue1 += sA1[threadIdx.y][j] * sB1[j][threadIdx.x];
            }
            __syncthreads();
        }
        if (Row < numCRows && Col < numCColumns){
            fa[Row*numCColumns + Col] = __expf(Cvalue1);
        }
    }
""")
func3 = mod.get_function("matrixMultiplyShared")

start.record()
g1 = g1.astype(np.float32)
h1 = h1.astype(np.float32)
i1 = np.zeros((g1.shape[0], h1.shape[1]))
i1 = i1.astype(np.float32)

matMul1 = np.empty_like(i1)
mults = 5000
for _ in range(mults):
    g_gpu1 = cuda.mem_alloc(g1.nbytes)
    cuda.memcpy_htod(g_gpu1, g1)
    h_gpu1 = cuda.mem_alloc(h1.nbytes)
    cuda.memcpy_htod(h_gpu1, h1)
    i_gpu1 = cuda.mem_alloc(i1.nbytes)
    cuda.memcpy_htod(i_gpu1, i1)

    func3(g_gpu1, h_gpu1, i_gpu1, block=(Tile_size, Tile_size, 1), grid=((i1.shape[1] // Tile_size)
    + 1, (i1.shape[0] // Tile_size) + 1, 1))
    # func3(g_gpu1, h_gpu1, i_gpu1, h_gpu2, i_gpu2, h_gpu3, i_gpu3, h_gpu4, i_gpu4, prev_cell_matrix_gpu,
    #      cell_memory_matrix_gpu, activation_matrix_gpu, block=dimBlock, grid=dimGrid)
    cuda.memcpy_dtoh(matMul1, i_gpu1)

end.record()
end.synchronize()

millis = start.time_till(end)
print("--- %s GPU Data Transfer seconds ---" % (millis / 1000))

start_time = time.time()
for _ in range(mults):
    C1 = np.exp(np.matmul(g1, h1), dtype=np.float64)

Time = time.time() - start_time
print("--- %s x100 CPU Matmul seconds ---" % (Time))

print(np.allclose(C1, matMul1, 1e-04, 1e-05))
print(np.amax(matMul1 - C1))

