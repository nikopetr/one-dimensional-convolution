# WARNING: Runs with Python 3.7.5. on the exported environment.
import random

# The functions that we implemented for the program
from MyFunctions import MyConvolveOld, MyConvolve

N = int(input("Give size N: "))
while N <= 10:
    N = int(input("Give size N (must be bigger than 10): "))

# Generating random N numbers for A from the range  [-1000. 1000].
A = []
for i in range(0, N):
    A.append(random.randint(-10000, 10000) + round(random.random(), 4))

B = [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]


# Prints the lists A, B and C , where C represents the calculated convolution of A with B.
print("A: ", A)
print("B: ", B)
print("C: ", MyConvolve(A, B))