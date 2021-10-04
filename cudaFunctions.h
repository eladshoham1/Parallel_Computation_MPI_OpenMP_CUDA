#ifndef __CUDA_FUNCTIONS_H__
#define __CUDA_FUNCTIONS_H__

enum size { N = 256 };

int calculateHistogramCuda(int* numbers, int* histogram, int size);

#endif