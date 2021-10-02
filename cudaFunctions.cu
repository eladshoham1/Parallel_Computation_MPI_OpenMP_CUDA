#include <cuda_runtime.h>
#include <stdio.h>

#include "cudaFunctions.h"

enum size { THREADS_PER_BLOCK = 256 };

__global__ void fillHistogramByZero(int* histogram, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= size)
        return;

    histogram[id] = 0;
}

__global__ void calculateHistogram(int* numbers, int* histogram, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int sharedHistogram[THREADS_PER_BLOCK];

    if (id >= size)
        return;

    atomicAdd(sharedHistogram + numbers[id], 1);
    __syncthreads();
    
    histogram[numbers[id]] += sharedHistogram[numbers[id]]; 
}

void checkStatus(cudaError_t cudaStatus, int* numbers, int* histogram, const char* err)
{
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "%s\n", err);
        free(numbers);
        free(histogram);
        exit(EXIT_FAILURE);
    }
}

int calculateHistogramCuda(int* numbers, int* histogram, int size)
{
    int *devNumbers = 0, *devHistogram = 0;
    int blocksPerGrid = (size + THREADS_PER_BLOCK) / THREADS_PER_BLOCK;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void **)&devNumbers, size * sizeof(int));
    checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda malloc failed!");

    cudaStatus = cudaMalloc((void **)&devHistogram, THREADS_PER_BLOCK * sizeof(int));
    checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda malloc failed!");

    cudaStatus = cudaMemcpy(devNumbers, numbers, size, cudaMemcpyHostToDevice);
    checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda memcpy failed!");

    fillHistogramByZero<<<blocksPerGrid, THREADS_PER_BLOCK>>>(devHistogram, THREADS_PER_BLOCK);
    cudaStatus = cudaDeviceSynchronize();
    checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda kernel failed!");
    
    calculateHistogram<<<blocksPerGrid, THREADS_PER_BLOCK>>>(devNumbers, devHistogram, size);
    cudaStatus = cudaDeviceSynchronize();
    checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda kernel failed!");

    cudaStatus = cudaMemcpy(histogram, devHistogram, THREADS_PER_BLOCK, cudaMemcpyDeviceToHost);
    checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda memcpy failed!");

    cudaStatus = cudaFree(devNumbers);
    checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda free failed!");

    cudaStatus = cudaFree(devHistogram);
    checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda free failed!");

    return EXIT_SUCCESS;
}