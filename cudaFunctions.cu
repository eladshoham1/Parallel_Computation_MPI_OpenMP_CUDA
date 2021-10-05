#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <iostream>
using namespace std;

#include "cudaFunctions.h"

__global__ void calculateHistogram(int* numbers, int* histogram, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int sharedHistogram[N];
    
    sharedHistogram[threadIdx.x] = 0;
    __syncthreads();

    if (id < size)
        atomicAdd(&(sharedHistogram[numbers[id]]), 1);
    __syncthreads();

    atomicAdd(&histogram[threadIdx.x], sharedHistogram[threadIdx.x]);
}

int checkStatus(cudaError_t cudaStatus, int* numbers, int* histogram, string err)
{
    if (cudaStatus != cudaSuccess)
    {
        cout << err << endl;

        cudaFree(numbers);
        cudaFree(histogram);

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int calculateHistogramCuda(int* numbers, int* histogram, int size)
{
    int *devNumbers = 0, *devHistogram = 0;
    int threadsPerBlock = N;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&devNumbers, size * sizeof(int));
    if (checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda malloc failed!") == EXIT_FAILURE)
        return EXIT_FAILURE;

    cudaStatus = cudaMalloc((void**)&devHistogram, threadsPerBlock * sizeof(int));
    if (checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda malloc failed!") == EXIT_FAILURE)
        return EXIT_FAILURE;

    cudaStatus = cudaMemcpy(devNumbers, numbers, size, cudaMemcpyHostToDevice);
    if (checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda memcpy failed!") == EXIT_FAILURE)
        return EXIT_FAILURE;

    cudaStatus = cudaMemset(devHistogram, 0, N * sizeof(int));
    if (checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda memset failed!") == EXIT_FAILURE)
        return EXIT_FAILURE;

    calculateHistogram<<<blocksPerGrid, threadsPerBlock>>>(devNumbers, devHistogram, size);
    cudaStatus = cudaDeviceSynchronize();
    if (checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda kernel failed!") == EXIT_FAILURE)
        return EXIT_FAILURE;

    cudaStatus = cudaMemcpy(histogram, devHistogram, threadsPerBlock, cudaMemcpyDeviceToHost);
    if (checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda memcpy failed!") == EXIT_FAILURE)
        return EXIT_FAILURE;

    cudaStatus = cudaFree(devNumbers);
    if (checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda free failed!") == EXIT_FAILURE)
        return EXIT_FAILURE;

    cudaStatus = cudaFree(devHistogram);
    if (checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda free failed!") == EXIT_FAILURE)
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}