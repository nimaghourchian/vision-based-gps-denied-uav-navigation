from multiprocessing import Lock, shared_memory

import numpy as np


class SharedArrayWithLock:
    """Small wrapper around a named NumPy array in multiprocessing shared memory."""

    def __init__(self, name, shape, dtype=np.float64, create=True):
        self.name = name
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.lock = Lock()
        self.size = int(np.prod(shape) * self.dtype.itemsize)

        if create:
            self.shm = shared_memory.SharedMemory(
                name=name, create=True, size=self.size
            )
            self.array = np.ndarray(shape, dtype=self.dtype, buffer=self.shm.buf)
            with self.lock:
                self.array[:] = np.nan
        else:
            self.shm = shared_memory.SharedMemory(name=name)
            self.array = np.ndarray(shape, dtype=self.dtype, buffer=self.shm.buf)

    def read(self):
        with self.lock:
            return self.array.copy()

    def write(self, data):
        with self.lock:
            np.copyto(self.array, data)

    def get(self):
        """Return direct array access; callers should hold the associated lock."""
        return self.array

    def close(self):
        self.shm.close()

    def unlink(self):
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass
