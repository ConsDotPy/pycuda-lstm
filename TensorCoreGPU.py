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
    return 1 / (1 + np.exp(-x, dtype=np.float64))


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

Tile_size = 16

m = 512
k = 1024
n = 512

g1 = np.random.randn(m, k)
h1 = np.random.randn(k, n)
h2 = np.random.randn(k, n)
h3 = np.random.randn(k, n)
h4 = np.random.randn(k, n)

mod1 = SourceModule("""
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

mod2 = SourceModule("""
        __global__ void matrixMultiplyToday(float *A, float *B, float *C){ 
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
            C[Row*numCColumns + Col] = Cvalue1;
        }
    }
""")

mod = SourceModule(
    """
    #include <mma.h>
    #include <cuda_fp16.h>
    using namespace nvcuda;
    extern "C" __global__ void wmma_example_gates(half *A1, half *B1, float *fa, half *B2, float *ia, half *B3, 
                                                  float *oa, half *B4, float *ga, float *prev_cell_matrix, 
                                                  float *cell_memory_matrix, float *activation_matrix) {
        // Must be multiples of 16 for wmma code to work
        int M =""" + str(m) + """;
        int K =""" + str(k) + """;
        int N =""" + str(n) + """;
        
        // The only dimensions currently supported by WMMA
        const int WMMA_M = 16;
        const int WMMA_N = 16;
        const int WMMA_K = 16;
        
       // Leading dimensions. Packed with no transpositions.
       int lda = M;
       int ldb = K;
       int ldc = N;
       
        float alpha = 1.0f;
        float beta = 1.0f;
        
       int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
       int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
       
       wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> pcm;
       wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cmm;
       wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> am;
       
       // Declare the fragments
       wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag1;
       wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag1;
       wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag1;
       wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag1;
       
              // Declare the fragments
       wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag2;
       wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag2;
       wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag2;
       
              // Declare the fragments
       wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag3;
       wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag3;
       wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag3;
       
              // Declare the fragments
       wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag4;
       wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag4;
       wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag4;
    
       wmma::fill_fragment(acc_frag1, 0.0f);
       wmma::fill_fragment(acc_frag2, 0.0f);
       wmma::fill_fragment(acc_frag3, 0.0f);
       wmma::fill_fragment(acc_frag4, 0.0f);
    
       // Loop over k
       for (int i = 0; i < K; i += WMMA_K) {
          int aRow = warpM * WMMA_M;
          int aCol = i;
    
          int bRow = i;
          int bCol = warpN * WMMA_N;
    
          // Bounds checking
          if (aRow < M && aCol < K && bRow < K && bCol < N) {
          
             // Load the inputs
             wmma::load_matrix_sync(b_frag1, A1 + bRow + bCol * ldb, ldb);
             wmma::load_matrix_sync(a_frag1, B1 + aRow + aCol * lda, lda);
             
             wmma::load_matrix_sync(a_frag2, B2 + aRow + aCol * lda, lda);
             
             wmma::load_matrix_sync(a_frag3, B3 + aRow + aCol * lda, lda);
             
             wmma::load_matrix_sync(a_frag4, B4 + aRow + aCol * lda, lda);
     
             // Perform the matrix multiplication
             wmma::mma_sync(acc_frag1, a_frag1, b_frag1, acc_frag1);
             wmma::mma_sync(acc_frag2, a_frag2, b_frag1, acc_frag2);
             wmma::mma_sync(acc_frag3, a_frag3, b_frag1, acc_frag3);
             wmma::mma_sync(acc_frag4, a_frag4, b_frag1, acc_frag4);
          }
       }
    
       // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
       int cRow = warpM * WMMA_M;
       int cCol = warpN * WMMA_N;
    
       if (cRow < M && cCol < N) {
          wmma::load_matrix_sync(c_frag1, fa + cRow + cCol * ldc, ldc, wmma::mem_col_major);
          wmma::load_matrix_sync(c_frag2, ia + cRow + cCol * ldc, ldc, wmma::mem_col_major);
          wmma::load_matrix_sync(c_frag3, oa + cRow + cCol * ldc, ldc, wmma::mem_col_major);
          wmma::load_matrix_sync(c_frag4, ga + cRow + cCol * ldc, ldc, wmma::mem_col_major);
          
          wmma::load_matrix_sync(pcm, prev_cell_matrix + cRow + cCol * ldc, ldc, wmma::mem_col_major);
          wmma::load_matrix_sync(cmm, cell_memory_matrix + cRow + cCol * ldc, ldc, wmma::mem_col_major);
          wmma::load_matrix_sync(am, activation_matrix + cRow + cCol * ldc, ldc, wmma::mem_col_major);
          
          for(int i=0; i < c_frag1.num_elements; i++) {
             c_frag1.x[i] = (float) alpha * __fdividef(1.0, 1 + __expf(-1*acc_frag1.x[i])) + beta * c_frag1.x[i];
             c_frag2.x[i] = (float) alpha * __fdividef(1.0, 1 + __expf(-1*acc_frag2.x[i])) + beta * c_frag2.x[i];
             c_frag3.x[i] = (float) alpha * __fdividef(1.0, 1 + __expf(-1*acc_frag3.x[i])); + beta * c_frag3.x[i];
             c_frag4.x[i] = (float) alpha * tanhf(acc_frag4.x[i]) + beta * c_frag4.x[i];
             cmm.x[i] = c_frag1.x[i]*pcm.x[i] + c_frag2.x[i]*c_frag4.x[i];
             am.x[i] = c_frag3.x[i]*tanhf(cmm.x[i]);
             
          }
          
          // Store the output
          wmma::store_matrix_sync(fa + cRow + cCol * ldc, c_frag1, ldc, wmma::mem_col_major);
          wmma::store_matrix_sync(ia + cRow + cCol * ldc, c_frag2, ldc, wmma::mem_col_major);
          wmma::store_matrix_sync(oa + cRow + cCol * ldc, c_frag3, ldc, wmma::mem_col_major);
          wmma::store_matrix_sync(ga + cRow + cCol * ldc, c_frag4, ldc, wmma::mem_col_major);
          
          wmma::store_matrix_sync(prev_cell_matrix + cRow + cCol * ldc, pcm, ldc, wmma::mem_col_major);
          wmma::store_matrix_sync(cell_memory_matrix + cRow + cCol * ldc, cmm, ldc, wmma::mem_col_major);
          wmma::store_matrix_sync(activation_matrix + cRow + cCol * ldc, am, ldc, wmma::mem_col_major);
       }
    }
""", arch="sm_75", no_extern_c=True)
func3 = mod.get_function("wmma_example_gates")
func4 = mod1.get_function("matrixMultiplyShared")


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

prev_cell_matrix1 = np.random.randn(m, n)
prev_cell_matrix = prev_cell_matrix1.astype(np.float32)

cell_memory_matrix = np.zeros((m, n))
cell_memory_matrix = cell_memory_matrix.astype(np.float32)

activation_matrix = np.zeros((m, n))
activation_matrix = activation_matrix.astype(np.float32)

matMul1 = np.empty_like(i1)
matMul2 = np.empty_like(i2)
matMul3 = np.empty_like(i3)
matMul4 = np.empty_like(i4)

cmm = np.empty_like(i4)
activations = np.empty_like(i4)

# The only dimensions currently supported by WMMA
WMMA_M = 16
WMMA_N = 16
WMMA_K = 16

# blockDim.x must be a multple of warpSize
# 128x4 means we have 16 warps and a block computes a 64x64 output tile
blockDim_x = 128
blockDim_y = 4

gridDim_x = (m + (WMMA_M * blockDim_x // 32 - 1)) // (WMMA_M * blockDim_x // 32)
gridDim_y = (n + WMMA_N * blockDim_y - 1) // (WMMA_N * blockDim_y)

mults = 50
start.record()

for _ in range(mults):
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

    func4(g_gpu1, h_gpu1, i_gpu1, h_gpu2, i_gpu2, h_gpu3, i_gpu3, h_gpu4, i_gpu4, prev_cell_matrix_gpu,
          cell_memory_matrix_gpu, activation_matrix_gpu, block=(Tile_size, Tile_size, 1),
          grid=((i4.shape[1] // Tile_size) + 1, (i4.shape[0] // Tile_size) + 1, 1))

    cuda.memcpy_dtoh(matMul1, i_gpu1)
    cuda.memcpy_dtoh(matMul2, i_gpu2)
    cuda.memcpy_dtoh(matMul3, i_gpu3)
    cuda.memcpy_dtoh(matMul4, i_gpu4)

    cuda.memcpy_dtoh(cmm, cell_memory_matrix_gpu)
    cuda.memcpy_dtoh(activations, activation_matrix_gpu)

end.record()
end.synchronize()

millis = start.time_till(end)
print(("--- {0} x{1} GPU Shared Mults seconds ---").format((millis / 1000), mults))

start = cuda.Event()
end = cuda.Event()

g1 = np.random.randn(m, k)
h1 = np.random.randn(k, n)
h2 = np.random.randn(k, n)
h3 = np.random.randn(k, n)
h4 = np.random.randn(k, n)

g1 = g1.astype(np.half)
h1 = h1.astype(np.half)
i1 = np.zeros((g1.shape[0], h2.shape[1]))
i1 = i1.astype(np.float32)
h2 = h2.astype(np.half)
i2 = np.zeros((g1.shape[0], h2.shape[1]))
i2 = i2.astype(np.float32)
h3 = h3.astype(np.half)
i3 = np.zeros((g1.shape[0], h3.shape[1]))
i3 = i3.astype(np.float32)
h4 = h4.astype(np.half)
i4 = np.zeros((g1.shape[0], h4.shape[1]))
i4 = i4.astype(np.float32)

start.record()
for _ in range(mults):
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

    func3(g_gpu1, h_gpu1, i_gpu1, h_gpu2, i_gpu2, h_gpu3, i_gpu3, h_gpu4, i_gpu4, prev_cell_matrix_gpu,
          cell_memory_matrix_gpu, activation_matrix_gpu, block=(blockDim_x, blockDim_y, 1),
          grid=(gridDim_x, gridDim_y, 1))

    cuda.memcpy_dtoh(matMul1, i_gpu1)
    cuda.memcpy_dtoh(matMul2, i_gpu2)
    cuda.memcpy_dtoh(matMul3, i_gpu3)
    cuda.memcpy_dtoh(matMul4, i_gpu4)
    cuda.memcpy_dtoh(cmm, cell_memory_matrix_gpu)
    cuda.memcpy_dtoh(activations, activation_matrix_gpu)

end.record()
end.synchronize()

millis = start.time_till(end)
print(("--- {0} x{1} GPU Tensor Mults seconds ---").format((millis / 1000), mults))

start_time = time.time()
for _ in range(mults//25):
    fa = np.matmul(g1, h1)
    C1 = sigmoid(fa)
    ia = np.matmul(g1, h2)
    C2 = sigmoid(ia)
    oa = np.matmul(g1, h3)
    C3 = sigmoid(oa)
    ga = np.matmul(g1, h4)
    C4 = tanh_activation(ga)

    cemema = np.multiply(C1, prev_cell_matrix1) + np.multiply(C2, C4)
    activation_matrix = np.multiply(C3, tanh_activation(cemema))
print(fa.shape)
Time = time.time() - start_time
print(("--- {0} x{1} CPU Mults seconds ---").format(Time, mults//25))

print(np.allclose(C1, matMul1, 1e-02, 1e-04))
print(np.allclose(C2, matMul2, 1e-02, 1e-04))
print(np.allclose(C3, matMul3, 1e-02, 1e-04))
print(np.allclose(C4, matMul4, 1e-02, 1e-04))
print()
print(np.amax(np.abs(C1 - matMul1)))
print(np.amax(np.abs(C2 - matMul2)))
print(np.amax(np.abs(C3 - matMul3)))
print(np.amax(np.abs(C4 - matMul4)))
print()
print(np.allclose(cemema, cmm, 1e-01, 1e-04))
print(np.amax(cmm), np.amin(cmm))
print(np.amax(cemema), np.amin(cemema))
print(np.amax(np.abs(cemema - cmm)))
print(np.amax(np.abs(cemema - cmm)))
print()
print(np.allclose(activation_matrix, activations, 1e-01, 1e-04))
print(np.amax(activations), np.amin(activations))
print(np.amax(activation_matrix), np.amin(activation_matrix))
print(np.amax(np.abs(activation_matrix - activations)))
print(np.amax(np.abs(activation_matrix - activations)))
