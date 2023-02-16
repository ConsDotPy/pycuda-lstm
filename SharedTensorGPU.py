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

m = 256*16
k = 256*16
n = 256*16

start_time = time.time()
# crear matrices para la GPU
g = np.random.randn(m, k)
g = g.astype(np.half)

h = np.random.randn(k, n)
h = h.astype(np.half)

i = np.zeros((m, n))
i = i.astype(np.float32)

j = np.zeros((m, n))
j = j.astype(np.float32)

matMul = np.empty_like(j)
print("--- %s CPU Create data seconds ---" % (time.time() - start_time))

# The only dimensions currently supported by WMMA
WMMA_M = 16
WMMA_N = 16
WMMA_K = 16
BLOCK_ROW_WARPS = 2
BLOCK_COL_WARPS = 4

WARP_ROW_TILES = 4
WARP_COL_TILES = 2

BLOCK_ROW_TILES = (WARP_ROW_TILES * BLOCK_ROW_WARPS)
BLOCK_COL_TILES = (WARP_COL_TILES * BLOCK_COL_WARPS)

M = 16
N = 16
K = 16
CHUNK_K = 4

SKEW_HALF = 16

SHMEM_SZ = np.amax(np.array([2 * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2, M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * 4]))
print(SHMEM_SZ)
mod = SourceModule(
    """
    #include <cuda.h>
    #include <mma.h>
    #include <stdio.h>
    using namespace nvcuda;
    // Externally configurable parameters.
 
    // Set this to 0 to use more than 64 Kb of shared memory to cache data, to
    // improve the performance of the computations on GPU.
    // Note that you need a GPU that can have more than 64 Kb of shared memory
    // per multiprocessor.
    #define SHARED_MEMORY_LIMIT_64K 1
    
    // GPU configuration.
    
    #define WARP_SIZE 32
    
    // MMA matrix tile dimensions.
    
    #define M 16
    #define N 16
    #define K 16
    
    #define WMMA_M 16
    #define WMMA_N 16
    #define WMMA_K 16
    
    // GEMM configuration.
    
    #define M_TILES 256
    #define N_TILES 256
    #define K_TILES 256
    
    #define M_GLOBAL (M * M_TILES)
    #define N_GLOBAL (N * N_TILES)
    #define K_GLOBAL (K * K_TILES)
    
    #define C_LAYOUT wmma::mem_row_major
    
    // Implementation constants.
    
    #define WARPS_PER_BLOCK 8
    #define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
    
    #if SHARED_MEMORY_LIMIT_64K
    // With only 64 Kb shared memory available, we can fit two 8-tile chunks of
    // the A and B matrix data, that are 16 * 16 * 8 * 8 * 2 = 32 Kb each
    // (i.e. two 8x8 arrays of tiles of 16x16 half-typed elements per CTA).
    // But we cannot account the 8 Kb total skew overhead, without which the
    // performance would be severely impacted. So we choose to reduce the chunk size
    // in half, i.e. the amount of A and B matrix data we cache in shared memory.
    // Accordingly, this doubles the number of outer iterations across the global K
    // dimension, which only slightly impacts the performance.
    #define CHUNK_K 4
    #else
    #define CHUNK_K 8
    #endif
    
    #define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(half))
    #define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
    #define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
    #define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)
    
    #define BLOCK_ROW_WARPS 2
    #define BLOCK_COL_WARPS 4
    
    #define WARP_ROW_TILES 4
    #define WARP_COL_TILES 2
    
    #define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
    #define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)
    
    #define GLOBAL_MEM_STRIDE N_GLOBAL
    
    #define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
    #define SHMEM_OFFSET (N * WARP_ROW_TILES)
    
    // The macro below is used to shift rows of the A matrix and columns of the B matrix
    // in shared memory to minimize possible bank conflicts.
    // Before performing the nvcuda::wmma::mma_sync operation, the warp must load the matrix
    // data using the nvcuda::wmma::load_matrix_sync operation. Although the memory access pattern
    // is not specified for that function, each lane in the warp can read one or multiple matrix
    // elements from different matrix rows or columns.
    // For shared memory, such access can result in bank conflicts if different rows / columns
    // of the matrix map to the same bank. By shifting each row and column by a few bytes, we
    // make sure that they map to different banks, thus reducing the number of possible bank
    // conflicts.
    // The number of 16 two-byte "half" elements is chosen as the minimum possible shift because
    // we must keep each row and column 256-bit aligned, as required by nvcuda::wmma::load_matrix_sync.
    #define SKEW_HALF 16
    
    extern "C" __global__ void compute_gemm(half *A, half *B, float *C, float *D) {
    
      float alpha = 1.0f;
      float beta = 1.0f;
      
      extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];
    
      // Warp and lane identification.
      const unsigned int warpId = threadIdx.x / WARP_SIZE;
      const unsigned int laneId = threadIdx.x % WARP_SIZE;
    
      // Offset in shared memory from which the B matrix is stored.
      const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;
    
      // This pointer is used to access the C and D matrix tiles this warp computes.
      float *shmem_warp_tile_ptr = (float *)&shmem[0][0] +
                                   (warpId / 2) * SHMEM_STRIDE * K * 2 +
                                   (warpId % 2) * SHMEM_OFFSET;
    
      // This pointer is used to stream the C and D matrices block-wide tile to and
      // from shared memory.
      float *shmem_warp_stream_ptr =
          (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * K;
    
      // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
      // each tile computation. Technically this is not generally correct (may
      // result in a loss of precision). Zero still needs to be specially handled
      // though.
      beta /= alpha;
    
      // Each CTA slides along the 128 x 128 tiles from the top left corner of the
      // matrix to the right and down, and selects the next tile to compute. Once
      // there's no such tile, all warps in this CTA exit.
      for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
        const unsigned int block_tile_i =
            ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
        const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;
    
        // Stop when there are no more D matrix tiles to compute in this CTA.
        if (block_tile_i >= M_TILES) {
          break;
        }
    
        // This warp's pointer to the C matrix data to copy memory from to shared
        // memory.
        const size_t gmem_idx =
            (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
        const float *src_gmem_warp_stream_ptr = &C[gmem_idx];
    
        // Stream multiple C tiles to shared memory.
    #pragma unroll
        for (int i = 0; i < K; i++) {
          typedef int4 copy_t;
    
          *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
              *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) +
                laneId);
        }
    
        __syncthreads();
    
        // These fragments will accumulate the result of A and B matrix fragment
        // multiplications along the K_GLOBAL dimension.
        wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES]
                                                           [WARP_ROW_TILES];
    
        // Load the C matrix tiles into fragments from shared memory.
    #pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
    #pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            const float *tile_ptr =
                shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;
    
            wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
          }
        }
    
        __syncthreads();
    
        // Scale the C matrix.
    #pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
    #pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
    #pragma unroll
            for (int t = 0; t < c[i][j].num_elements; t++) {
              c[i][j].x[t] *= beta;
            }
          }
        }
    
        // Select what warp copies what matrix to shared memory.
        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        const half *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL] +
                                               M * K_GLOBAL * (warpId % 4) * 2)
                                            : (&B[block_tile_j * N * K_GLOBAL] +
                                               N * K_GLOBAL * (warpId % 4) * 2);
    
        // Go through the global K dimension by a fixed step at a time.
    #pragma unroll
        for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
          // Copy slices of the A and B matrices to shared memory.
          // The first half of the warps in the CTA copy the A matrix, the rest copy
          // the B matrix.
          size_t shmem_idx =
              warpId < (WARPS_PER_BLOCK / 2)
                  ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
                  : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);
    
          // First half of the warp copies the first row / column of the matrix,
          // the second half of the warp copies the next.
          int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * K +
                                    (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) +
                           (laneId % CHUNK_COPY_LINE_LANES);
    
          // Shift the second half of the warp to the next row / column in the
          // shared memory.
          shmem_idx += laneId / CHUNK_COPY_LINE_LANES;
    
    #pragma unroll
          for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
               i++) {
            // Copy 16 bytes at once in each lane.
            *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
                *lane_ptr;
    
            // Advance the global memory pointer and the shared memory index.
            lane_ptr =
                (int4 *)((half *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
            shmem_idx += CHUNK_COPY_LINES_PER_WARP;
          }
    
          __syncthreads();
    
          // Compute a grid of C matrix tiles in each warp.
    #pragma unroll
          for (int k_step = 0; k_step < CHUNK_K; k_step++) {
            wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major>
                a[WARP_COL_TILES];
            wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major>
                b[WARP_ROW_TILES];
    
    #pragma unroll
            for (int i = 0; i < WARP_COL_TILES; i++) {
              size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
              const half *tile_ptr = &shmem[shmem_idx_a][k_step * K];
    
              wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);
    
    #pragma unroll
              for (int j = 0; j < WARP_ROW_TILES; j++) {
                if (i == 0) {
                  // Load the B matrix fragment once, because it is going to be
                  // reused against the other A matrix fragments.
                  size_t shmem_idx_b = shmem_idx_b_off +
                                       (WARP_ROW_TILES * N) * (warpId % 2) +
                                       (j * N);
                  const half *tile_ptr = &shmem[shmem_idx_b][k_step * K];
    
                  wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF);
                }
    
                wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
              }
            }
          }
    
          __syncthreads();
        }
    
          // Store the D fragments to shared memory.
    #pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
    #pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
    #pragma unroll
            // Uniform, point-wise transformations of ALL fragment elements by ALL
            // threads in the warp are well-defined even though element indices
            // within fragment storage are not defined.
            for (int t = 0; t < c[i][j].num_elements; t++) c[i][j].x[t] *= alpha;
    
            float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;
    
            wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
          }
        }
    
        __syncthreads();
    
        // Now that shared memory contains all the D tiles, stream them to global
        // memory.
        float *dst_gmem_warp_stream_ptr = &D[gmem_idx];
    
    #pragma unroll
        for (int i = 0; i < K; i++) {
          *((int4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
              *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
        }
    
        __syncthreads();
      }
    }
    """, arch="sm_75", no_extern_c=True)

# Instaciar kernel
func = mod.get_function("compute_gemm")


start = cuda.Event()
end = cuda.Event()
start.record()

mults = 1

for _ in range(mults):
    g_gpu = cuda.mem_alloc(g.nbytes)
    cuda.memcpy_htod(g_gpu, g)
    h_gpu = cuda.mem_alloc(h.nbytes)
    cuda.memcpy_htod(h_gpu, h)
    i_gpu = cuda.mem_alloc(i.nbytes)
    cuda.memcpy_htod(i_gpu, i)
    j_gpu = cuda.mem_alloc(j.nbytes)
    cuda.memcpy_htod(j_gpu, j)
    func(g_gpu, h_gpu, i_gpu, j_gpu, block=(32*8, 1, 1), grid=(20, 1))
    cuda.memcpy_dtoh(matMul, j_gpu)
end.record()
end.synchronize()

millis = start.time_till(end)
print("--- {0} x{1} GPU Matmul seconds ---".format((millis / 1000), str(mults)))

start_time = time.time()
for _ in range(mults):
    hostMult = np.matmul(g, h)
print("--- {0} x{1} CPU Matmul seconds ---".format((time.time() - start_time), str(mults)))

# Validar diferencia absoluta
print(matMul.shape, hostMult.shape)
print(np.allclose(matMul.T, hostMult, 1e-1, 1e-1))
print(np.amax(matMul), np.amin(matMul))
print(np.amax(hostMult), np.amin(hostMult))
print(np.amax(abs(matMul.T - hostMult)))