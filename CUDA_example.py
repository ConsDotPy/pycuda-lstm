import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from sys import getsizeof

n = 35
m = 20
o = 25

a = np.random.randn(n, m)
a = a.astype(np.float32)
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

b = np.random.randn(m, 1)
b = b.astype(np.float32)
b_gpu = cuda.mem_alloc(b.nbytes)
cuda.memcpy_htod(b_gpu, b)

c = np.zeros((n, 1))
c = c.astype(np.float32)
c_gpu = cuda.mem_alloc(c.nbytes)
cuda.memcpy_htod(c_gpu, c)

d = np.random.randn(n, m)
d = d.astype(np.float64)
d_gpu = cuda.mem_alloc(d.nbytes)
cuda.memcpy_htod(d_gpu, d)

e = np.random.randn(n, m)
e = e.astype(np.float64)
e_gpu = cuda.mem_alloc(e.nbytes)
cuda.memcpy_htod(e_gpu, e)

f = np.random.randn(n, m)
f = f.astype(np.float64)
f_gpu = cuda.mem_alloc(f.nbytes)
cuda.memcpy_htod(f_gpu, f)

g = np.random.randn(m, n)
g = g.astype(np.float64)
g_gpu = cuda.mem_alloc(g.nbytes)
cuda.memcpy_htod(g_gpu, g)

h = np.random.randn(n, o)
h = h.astype(np.float64)
h_gpu = cuda.mem_alloc(h.nbytes)
cuda.memcpy_htod(h_gpu, h)

i = np.zeros((g.shape[0], h.shape[1]))
i = i.astype(np.float64)
i_gpu = cuda.mem_alloc(i.nbytes)
cuda.memcpy_htod(i_gpu, i)

mod = SourceModule('\n'
                    '__global__ void gpu_matrix_mult(double *a, double *b, double *c) {'
                        'int m = '+str(g.shape[0])+';'
                        'int n = '+str(h.shape[0])+';'
                        'int k = '+str(h.shape[1])+';'
                        'int row = blockIdx.y * blockDim.y + threadIdx.y;'
                        'int col = blockIdx.x * blockDim.x + threadIdx.x;'
                        'double sum = 0.0;'
                        'if( col < k && row < m) {'
                            'for(int i = 0; i < n; i++) {'
                                'sum += a[row * n + i] * b[i * k + col];'
                            '}'
                            'c[row * k + col] = sum;'
                        '}'
                    '}'
                   "__global__ void matVec_mult (float *a, float *b, float *c) {\n"
                   "    int n =" + str(n) + ";\n"
                   "int m =" + str(m) + ";\n"
                   "int row = blockDim.x * blockIdx.x + threadIdx.x;\n "
                   "if(row < n) {\n"
                   "  for(int i = 0; i < m; i++) {\n"
                   "      c[row] = c[row] + a[m*row + i] * b[i];\n"
                   "        }\n"
                   "    }\n"
                   "}\n"
                   "__global__ void hada_mult (double *a, double *b, double *c) {\n "
                   "    int row =" + str(n) + ";"
                   "    int col =" + str(m) + ";"
                   "    int x = blockIdx.x * blockDim.x + threadIdx.x;\n "
                   "    int y = blockIdx.y * blockDim.y + threadIdx.y;\n "
                   "   if (x < col && y < row ) {\n"
                   "       c[y*x] = a[y*x] * b[y*x];\n"
                   "   }\n"
                   "}")

func1 = mod.get_function("matVec_mult")
func1(a_gpu, b_gpu, c_gpu, block=(1, 1, 1), grid=(512, 1))

func2 = mod.get_function("hada_mult")
func2(d_gpu, e_gpu, f_gpu, block=(1, 1, 1), grid=(512, 1))

func3 = mod.get_function("gpu_matrix_mult")
func3(g_gpu, h_gpu, i_gpu, block=(2, 2, 1), grid=(h.shape[1], g.shape[0]))

hadamard = np.empty_like(f)
matVec = np.empty_like(c)
matMul = np.empty_like(i)

cuda.memcpy_dtoh(matVec, c_gpu)
cuda.memcpy_dtoh(hadamard, f_gpu)
cuda.memcpy_dtoh(matMul, i_gpu)

# print('A\n', a)
# print('B\n', b)
# print('GPU C = A x B\n', matVec)
hostMult = np.matmul(a, b)
print(np.array_equal(matVec.astype(int), hostMult.astype(int)))
# print('HOST C = A x B\n', hostMult)
# print('D\n', d)
# print('E\n', e)
print('GPU F = D o E\n', hadamard)
print('HOST F = D o E\n', np.multiply(d, e))
hostMult = np.multiply(d, e)
print(np.array_equal(hadamard.astype(int), hostMult.astype(int)))
hostMatMul = np.matmul(g, h)
print(np.array_equal(matMul.astype(int), hostMatMul.astype(int)))
