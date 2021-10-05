#include <cuda_runtime.h>
#include <iostream>
using namespace std;

#include "cudaFunctions.h"


__global__ void fillHistogramByZero(int* histogram, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id>=size)
        return;
    
    histogram[id] = 1;
}


__global__ void calculateHistogram(int* numbers, int* histogram, int size)
{
    __shared__ int sharedHistogram[N];
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x < N)
        sharedHistogram[threadIdx.x] = 64;
    __syncthreads();

    if (id < size)
        atomicAdd(&sharedHistogram[numbers[id]], 1);
    __syncthreads();

    if (threadIdx.x < N)
        // atomicAdd(&histogram[threadIdx.x],sharedHistogram[threadIdx.x]);
        histogram[threadIdx.x]=sharedHistogram[threadIdx.x];
}

void checkStatus(cudaError_t cudaStatus, int* numbers, int* histogram, string err)
{
    if (cudaStatus != cudaSuccess)
    {
        cout << err << endl;
        cudaFree(numbers);
        cudaFree(histogram);
        exit(EXIT_FAILURE);
    }
}

int calculateHistogramCuda(int* numbers, int* histogram, int size)
{
    int *devNumbers = 0, *devHistogram = 0;
    int threadsPerBlock = N;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&devNumbers, size * sizeof(int));
    checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda malloc failed!");

    cudaStatus = cudaMalloc((void**)&devHistogram, threadsPerBlock * sizeof(int));
    checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda malloc failed!");

    cudaStatus = cudaMemcpy(devNumbers, numbers, size, cudaMemcpyHostToDevice);
    checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda memcpy failed!");

    // fillHistogramByZero<<<blocksPerGrid, threadsPerBlock>>>(devHistogram, N);
    // cudaStatus = cudaDeviceSynchronize();
    // checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda kernel failed!");

    calculateHistogram<<<blocksPerGrid, threadsPerBlock>>>(devNumbers, devHistogram, size);
    cudaStatus = cudaDeviceSynchronize();
    checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda kernel failed!");

    cudaStatus = cudaMemcpy(histogram, devHistogram, threadsPerBlock, cudaMemcpyDeviceToHost);
    checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda memcpy failed!");

    cudaStatus = cudaFree(devNumbers);
    checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda free failed!");

    cudaStatus = cudaFree(devHistogram);
    checkStatus(cudaStatus, devNumbers, devHistogram, "Cuda free failed!");

    return EXIT_SUCCESS;
}