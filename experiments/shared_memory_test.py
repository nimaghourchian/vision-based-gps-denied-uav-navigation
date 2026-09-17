from multiprocessing import shared_memory

import numpy as np


shm = shared_memory.SharedMemory(create=True, size=3 * 5 * 4)
array = np.ndarray(shape=(3, 5), dtype=np.float32, buffer=shm.buf)
array[:] = 42.0
print(array)

shm.close()
shm.unlink()
