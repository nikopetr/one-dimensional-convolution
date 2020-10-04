# WARNING: Runs with Python 3.7.5. on the exported environment.
import random
import time

from MyFunctions import MyConvolve, MyConvolveCUDA # The functions that we implemented for the program

# Comparison between CPU and GPU for Convolution:

N = int(input("Give size N: "))
while N <= 10:
    N = int(input("Give size N (must be bigger than 10): "))

# Generating random N numbers for A from the range  [-1000. 1000].
A = []
for i in range(0, N):
    A.append(random.randint(-10000, 10000) + round(random.random(), 4))

B = [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]

print("Time comparison between CPU and GPU for Convolution for input: N =", N)
start_time = time.time()  # Records the time that the convolution procedure has started.
print("C: ", MyConvolve(A, B))
print("Calculations on CPU:  %s seconds" % (time.time() - start_time))  # Calculates and prints the total time that the convolution as taken

start_time = time.time()  # Records the time that the convolution procedure has started.
print("C: ", MyConvolveCUDA(A, B))
print("Calculations on GPU with CUDA (Using user-defined raw Kernel):  %s seconds" % (time.time() - start_time))  # Calculates and prints the total time that the convolution as taken
