import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from sys import getsizeof
import time


# info de la GPU

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


start = cuda.Event()
end = cuda.Event()

Tile_size = 16

m = 128
n = 128
o = 128

start_time = time.time()
# crear matrices para la GPU
g = np.random.randn(m, n)
g = g.T.astype(np.half)

h = np.random.randn(n, o)
h = h.T.astype(np.half)

i = np.zeros((g.shape[0], h.shape[1]))
i = i.astype(np.float32)

matMul = np.empty_like(i)
print("--- %s CPU Create data seconds ---" % (time.time() - start_time))

# The only dimensions currently supported by WMMA
WMMA_M = 16
WMMA_N = 16
WMMA_K = 16

mod = SourceModule(
    """
    #include <mma.h>
    #include <cuda_fp16.h>
    using namespace nvcuda;

    extern "C" __global__ void wmma_example_Shrd(half *a, half *b, float *c) {
        // Must be multiples of 16 for wmma code to work
        int M =""" + str(m) + """;
        int N =""" + str(n) + """;
        int K =""" + str(o) + """;

        // The only dimensions currently supported by WMMA
        const int WMMA_M = 16;
        const int WMMA_N = 16;
        const int WMMA_K = 16;

       // Leading dimensions. Packed with no transpositions.
       int lda = M;
       int ldb = K;
       int ldc = M;

        float alpha = 1.0f;
        float beta = 1.0f;

       // Tile using a 2D grid
       int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
       int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

       // Declare the fragments
       wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
       wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
       wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
       wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

       wmma::fill_fragment(acc_frag, 0.0f);

       // Loop over k
       for (int i = 0; i < K; i += WMMA_K) {
          int aRow = warpM * WMMA_M;
          int aCol = i;

          int bRow = i;
          int bCol = warpN * WMMA_N;

          // Bounds checking
          if (aRow < M && aCol < K && bRow < K && bCol < N) {
             // Load the inputs
             wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
             wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

             // Perform the matrix multiplication
             wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

          }
       }

       // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
       int cRow = warpM * WMMA_M;
       int cCol = warpN * WMMA_N;

       if (cRow < M && cCol < N) {
          wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);


          for(int i=0; i < c_frag.num_elements; i++) {
             c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
          }

          // Store the output
          wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
       }
    }
""", arch="sm_75", no_extern_c=True, include_dirs=["-lcublas"])

# Instaciar kernel
func = mod.get_function("wmma_example_Shrd")

# blockDim.x must be a multple of warpSize
# 128x4 means we have 16 warps and a block computes a 64x64 output tile
blockDim_x = 128
blockDim_y = 4

gridDim_x = (m + (WMMA_M * blockDim_x // 32 - 1)) // (WMMA_M * blockDim_x // 32)
gridDim_y = (n + WMMA_N * blockDim_y - 1) // (WMMA_N * blockDim_y)

start = cuda.Event()
end = cuda.Event()
start.record()

mults = 5

for _ in range(mults):
    g_gpu = cuda.mem_alloc(g.nbytes)
    cuda.memcpy_htod(g_gpu, g)
    h_gpu = cuda.mem_alloc(h.nbytes)
    cuda.memcpy_htod(h_gpu, h)
    i_gpu = cuda.mem_alloc(i.nbytes)
    cuda.memcpy_htod(i_gpu, i)
    func(g_gpu, h_gpu, i_gpu, block=(blockDim_x, blockDim_y, 1), grid=(gridDim_x, gridDim_y, 1))
    cuda.memcpy_dtoh(matMul, i_gpu)
end.record()
end.synchronize()

millis = start.time_till(end)
print("--- {0} x{1} GPU Matmul seconds ---".format((millis / 1000), str(mults)))

start_time = time.time()
for _ in range(mults):
    hostMult = np.matmul(g, h)
print("--- {0} x{1} CPU Matmul seconds ---".format((time.time() - start_time), str(mults)))

# Validar diferencia absoluta
print(np.allclose(matMul.T, hostMult, 1e-1, 1e-1))
print(matMul.shape, hostMult.shape)
print(np.amax(matMul), np.amin(matMul))
print(np.amax(hostMult), np.amin(hostMult))
print(np.amax(abs(matMul.T - hostMult)))
print(np.amax(abs(matMul - hostMult)))