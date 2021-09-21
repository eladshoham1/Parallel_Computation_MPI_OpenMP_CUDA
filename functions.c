#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "functions.h"

int* readNumbers(int* size)
{
    int i, *numbers;

    scanf("%d", size);
    numbers = (int*)doMalloc(*size * sizeof(int));

    for (i = 0; i < *size; i++)
        scanf("%d", &numbers[i]);

    return numbers;
}

void printHistogram(int *counters, int size)
{
    int i;

    for (i = 0; i < size; i++)
    {
        if (counters[i] != 0)
            printf("%d: %d\n", i, counters[i]);
    } 
}

void calculateHistogramOpenMp(int* numbers, int* counters, int size)
{
#pragma omp parallel for
    for (int i = 0; i < size; i++)
        counters[numbers[i]]++;
}

void mergeHistogram(int* masterCounters, int* workerCounters, int size)
{
    int i;

    for (i = 0; i < size; i++)
        masterCounters[i] += workerCounters[i];
}

void* doMalloc(unsigned int nbytes) 
{
    void *p = malloc(nbytes);

    if (p == NULL) { 
        fprintf(stderr, "malloc failed\n"); 
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    return p;
}