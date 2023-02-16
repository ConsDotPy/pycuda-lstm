import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from sys import getsizeof
import time

start = cuda.Event()
end = cuda.Event()

m = 512
n = 512

g = np.random.randn(m, n)
g = g.astype(np.float32)
g_gpu = cuda.mem_alloc(g.nbytes)

h = np.random.randn(m, n)
h = h.astype(np.float32)
h_gpu = cuda.mem_alloc(h.nbytes)

i = np.zeros((g.shape[0], g.shape[1]))
i = i.astype(np.float32)
i_gpu = cuda.mem_alloc(i.nbytes)

matMul = np.empty_like(i)
Tile_size1 = 32

start.record()
cuda.memcpy_htod(g_gpu, g)
cuda.memcpy_htod(h_gpu, h)
cuda.memcpy_htod(i_gpu, i)

mod = SourceModule("""
__global__ void hada_mult(float *A, float *B, float *C)
{   
    int height = """ + str(g.shape[0]) + """;
    
    // ID
    int Row = blockDim.y*blockIdx.y + threadIdx.y;
    int Col = blockDim.x*blockIdx.x + threadIdx.x;
    
    // Mult
    C[Row * height + Col ] = A[Row * height + Col] * B[Row * height + Col];
}
"""
                   )
end.record()
end.synchronize()
millis = start.time_till(end)
print("--- %s seconds Data transfer ---" % (millis/1000))

start = cuda.Event()
end = cuda.Event()

start.record()
func3 = mod.get_function("hada_mult")
func3(g_gpu, h_gpu, i_gpu, block=(Tile_size1, Tile_size1, 1),
      grid=(g.shape[0]//Tile_size1+1, g.shape[1]//Tile_size1+1, 1))
cuda.memcpy_dtoh(matMul, i_gpu)
func3(g_gpu, h_gpu, i_gpu, block=(Tile_size1, Tile_size1, 1),
      grid=(g.shape[0]//Tile_size1+1, g.shape[1]//Tile_size1+1, 1))
cuda.memcpy_dtoh(matMul, i_gpu)
end.record()
end.synchronize()
millis = start.time_till(end)
print("--- %s seconds GPU & GPUtoHOST ---" % (millis/1000))

start_time = time.time()
hostMult = np.multiply(g, h)
print("--- %s seconds HOST ---" % (time.time() - start_time))
print(np.allclose(matMul, hostMult, 1e-05, 1e-08))
