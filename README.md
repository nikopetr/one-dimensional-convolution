# one-dimensional-convolution
1-D convolution implementation using Python and CUDA, implemented as a Signals and Systems university project.

This work in the Systems Signals course deals with the implementation of convolution algorithms where they also run on an Nvidia graphics card with the help of CUDA in a Python environment.
Implemented using Python version 3.7.5.
To make it easier for you to use the libraries I have included to run the program, I encourage you to import the environment file included through the Anaconda software. A file on how to import and run a project through Anaconda is also included.

## Convolution Algorithm Used
In order to avoid using the O(n^2) algorithm of the original definition, the method used is described as below:

It is known that another way to get the convolution of two signals is to first calculate the Fourier transform of each signal, and then their product will lead to the transformation of the requested convolution. Finally with a calculation of the inverse Fourier we will get the output of the convolution is needed.
Through fast algorithms for calculating the Fourier transform of a discrete sequence (eg Cooley-Tukey), we can calculate the transformation with time complexity of O(nlogn).
With this method the calculation of the a convolution algorithm totally takes O(nlogn), since we will essentially need to do the transformation three times and a simple element-by-element multiplication.

## The python files
The first task requires the generation of a random float array which is then convoluted with: `[0.2 0.2 0.2 0.2 0.2]`


## CPU / GPU comparison on execution time
A test was conducted with a vector of 8 000 000 random elements.

The two WAV files that were convoluted required about `13.3 minutes` in C++ and just `1.3 seconds` in CUDA.
