#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "myProto.h"

enum size { THREADS_PER_BLOCK = 256 };

__global__ void myKernel(int* numbers, int* tempCounters, int size)
{
  //__shared__ int *counters;
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < size)
      tempCounters[i]++;
}

int computeOnGPU(int* numbers, int* tempCounters, int size)
{
  cudaError_t err = cudaSuccess;
  int *numbersOnGpu, blocksPerGrid = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  err = cudaMalloc((void **)&numbersOnGpu, size * sizeof(int));
  if (err != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(numbersOnGpu, numbers, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
      fprintf(stderr, "Failed to copy data from host to device - %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  myKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(numbers, tempCounters, size);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(numbers, numbersOnGpu, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
      fprintf(stderr, "Failed to copy result array from device to host -%s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  if (cudaFree(numbersOnGpu) != cudaSuccess) {
      fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}