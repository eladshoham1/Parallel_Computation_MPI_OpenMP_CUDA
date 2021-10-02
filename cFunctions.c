#include <mpi.h>
#include <omp.h>
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

void histogramOpenMpReduction(int* numbers, int* histogram, int size)
{
#pragma omp parallel for //reduction(+: histogram)
	for (int i = 0; i < size; i++)
		localHistog[numbers[i]]++;
}

void histogramOpenMpPrivate(int* numbers, int** histograms, int size)
{
#pragma omp parallel for
    for (int i = 0; i < size; i++)
        histograms[omp_get_thread_num()][numbers[i]]++;
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