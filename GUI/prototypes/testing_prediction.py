import numpy as np
import time


class Rechner():
    def __init__(self):
        self.simtime = 10
        self.dt = 1
        self.total_iterations = int(self.simtime/self.dt)
        self.elapsed_time = 0
        self.remaining_time = 0

    def rechne(self):
        start_time = time.time()

        for iteration, t in enumerate(np.arange(0, self.simtime, self.dt)):
            for i in range(0, 10):
                print('test')

            self.elapsed_time = time.time() - start_time
            self.remaining_time = (self.total_iterations - iteration - 1) * self.elapsed_time / (iteration + 1)
            print(f"Iteration: {iteration + 1}/{self.total_iterations}. "
                  f"Elapsed time: {self.elapsed_time:.10f}s. Remaining time: {self.remaining_time:.10f}s")

r = Rechner()
r.rechne()
