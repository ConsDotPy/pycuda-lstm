import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from matplotlib import pyplot as plt

N = 500000

# Doble precisión
d = np.random.rand(N)

# Copia de precisión simple a CPU
f = d.astype(np.float32)

# Copia de precisión simple a GPU
g = gpuarray.to_gpu(f)

# Sumas
dsum = d.sum()
fsum = f.sum()
gsum = gpuarray.sum(g).get()

# Diferencia respecto a la doble precisión
ferr = np.abs(dsum-fsum)
gerr = np.abs(dsum-gsum)

print("Suma real:", dsum)
print("CPU Single", fsum, "Error:", ferr)
print("GPU Single", gsum, "Error:", gerr)

plt.bar(["CPU Error", "GPU Error"], [ferr, gerr], color="bg")
plt.show()
