#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "cFunctions.h"

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

void mergeHistogram(int* masterHistogram, int* workerHistogram, int size)
{
    int i;

    for (i = 0; i < size; i++)
        masterHistogram[i] += workerHistogram[i];
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