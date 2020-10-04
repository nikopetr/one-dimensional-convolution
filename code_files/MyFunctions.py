# WARNING: Runs with Python 3.7.5. on the exported environment.
import numpy
import cupy # Using cupy for GPU parallel programming

# This was the first idea of the implementation of the convolution, by using the original formula.
# Although the function works perfectly, the algorithm was not efficient for a big input since the complexity is O(n^2),
# that's why we came across a better approach later.
def MyConvolveOld(A, B):
    C = []
    lenC = len(A) + len(B) - 1 #Size of convolution

    for n in range(lenC):
        sum = 0
        for k in range(len(A)):
            if n - k >= 0 and len(B) > n - k:  # Den iparxei epikalipsi
                sum += A[k] * B[n - k] # Orismos Sineliksis

        C.append(sum)

    return C

# This is the better version for the implementation of the Convolution, which is calculated by taking the FFT (fast Fourier transform)
# of each input sequence, multiplying point-wise, and finally by performing an inverse FFT we will have the output of the Convolution.
# That way, by using the FFT algorithms the time complexity of the Convolution is reduced to O(N log N).
def MyConvolve(A, B):
    lenC = len(A) + len(B) - 1 # The size of the Convolution sequence

    ftA = numpy.fft.fft(A,lenC) # Calculates the FFT of the first input.
    ftB = numpy.fft.fft(B,lenC) # Calculates the FFT of the second input.

    ftC = numpy.multiply(ftA, ftB) # Calculates the FFT of the output by multiplying point-wise.(Leaves 0j at the end since it's a complex number list)
    C = numpy.fft.ifft(ftC).real # Since ftC elements will have 0J as an imaginary part we only want to take the real part from them.

    return C #Returns the output of the comvolution

myPointWiseMultKernel = cupy.RawKernel(r'''
#include <cupy/complex.cuh>
 extern "C" __global__ void my_func(const complex<float>* x1, const complex<float>* x2, complex<float>* y, const int N) {
   int tid = blockDim.x * blockIdx.x + threadIdx.x;
   if (tid < N)
     y[tid] = x1[tid] * x2[tid];
   }
   ''', name='my_func')

def MyConvolveCUDA(A, B):
    lenC = len(A) + len(B) - 1# The size of the Convolution sequence

    A = cupy.asarray(A)# To convert from numpy.ndarray to cupy.ndarray (moves the data to the gpu device)
    B = cupy.asarray(B)# To convert from numpy.ndarray to cupy.ndarray(moves the data to the gpu device)

    # Calculates the FFT of the output by multiplying point-wise.(Leaves 0j at the end since it's a complex number list)
    ftA = cupy.fft.fft(A, lenC).astype(dtype=cupy.complex64)# Calculates the FFT of the first input.
    ftB = cupy.fft.fft(B, lenC).astype(dtype=cupy.complex64)# Calculates the FFT of the second input.


    # Used own custom raw kernel for multiplication instead of cupy's multiply,
    threadsPerBlock = 512
    blocksPerGrid = (lenC + (threadsPerBlock - 1)) // threadsPerBlock

    ftC = cupy.zeros((lenC, ), dtype=cupy.complex64)
    myPointWiseMultKernel(grid=(blocksPerGrid,), block=(threadsPerBlock,), args=(ftA, ftB, ftC, lenC))  # grid, block and arguments
    #ftC = cupy.multiply(ftA, ftB) #Auto cupy's multiplication, can be used to compare the results.

    C = cupy.fft.ifft(ftC).real # Since ftC elements will have 0J as an imaginary part we only want to take the real part from them.
    return C #Returns the output of the comvolution