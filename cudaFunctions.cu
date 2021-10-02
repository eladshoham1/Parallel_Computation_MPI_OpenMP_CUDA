#include <cuda_runtime.h>
#include <iostream>
using namespace std;

#include "cudaFunctions.h"

enum size { THREADS_PER_BLOCK = 256 };

__global__ void calculateHistogram(int* numbers, int* histogram, int size)
{
    __shared__ int privateHistogram[THREADS_PER_BLOCK];
    int id = threadIdx.x;

    if (id < size)
        privateHistogram[id]++;
        
    __syncthreads();
}

void checkStatus(cudaError_t cudaStatus, int* numbers, std::string err)
{
    if (cudaStatus != cudaSuccess)
    {
        delete[] numbers;
        cout << err << endl;
        exit(EXIT_FAILURE);
    }
}

int calculateHistogramCuda(int* numbers, int* histogram, int size)
{
    int *numbersGpu, *histogramGpu;
    int blocksPerGrid = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void **)&numbersGpu, size * sizeof(int));
    checkStatus(cudaStatus, numbersGpu, "Cuda malloc failed!");

    cudaStatus = cudaMalloc((void **)&histogramGpu, THREADS_PER_BLOCK * sizeof(int));
    checkStatus(cudaStatus, numbersGpu, "Cuda malloc failed!");

    cudaStatus = cudaMemcpy(numbersGpu, numbers, size, cudaMemcpyHostToDevice);
    checkStatus(cudaStatus, numbersGpu, "Cuda memcpy failed!");

    cudaStatus = cudaMemcpy(histogramGpu, histogram, THREADS_PER_BLOCK, cudaMemcpyHostToDevice);
    checkStatus(cudaStatus, numbersGpu, "Cuda memcpy failed!");
    
    calculateHistogram<<<blocksPerGrid, THREADS_PER_BLOCK>>>(numbers, histogram, size);
    cudaStatus = cudaDeviceSynchronize();
    checkStatus(cudaStatus, numbersGpu, "Cuda kernel failed!");

    cudaStatus = cudaMemcpy(histogram, histogramGpu, THREADS_PER_BLOCK, cudaMemcpyDeviceToHost);
    checkStatus(cudaStatus, numbersGpu, "Cuda memcpy failed!");

    cudaStatus = cudaFree(numbersGpu);
    checkStatus(cudaStatus, numbersGpu, "Cuda free failed!");

    cudaStatus = cudaFree(histogramGpu);
    checkStatus(cudaStatus, numbersGpu, "Cuda free failed!");

    return EXIT_SUCCESS;
}