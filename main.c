#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "cFunctions.h"
#include "cudaFunctions.h"

int main(int argc, char *argv[])
{
    int *numbers, histogram[N] = { 0 }, workerHistogram[N] = { 0 };
    int size, halfSize, rank, numProcs, position = 0;
    char buff[BUFFER_SIZE];
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    omp_set_num_threads(4);

    if (rank == ROOT)
    {
        if (numProcs != 2) {
            printf("Run the example with two processes only\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        numbers = readNumbers(&size);
        halfSize = size / 2;

        MPI_Pack(&halfSize, 1 , MPI_INT, buff, BUFFER_SIZE, &position, MPI_COMM_WORLD);
        MPI_Pack(numbers + halfSize + size % 2, halfSize, MPI_INT, buff, BUFFER_SIZE, &position, MPI_COMM_WORLD);
        MPI_Send(buff, position, MPI_PACKED, WORKER ,0 ,MPI_COMM_WORLD);

        histogramOpenMpReduction(numbers, histogram, halfSize + size % 2);

        MPI_Recv(workerHistogram, N, MPI_INT, WORKER, 0, MPI_COMM_WORLD, &status);
        mergeHistogram(histogram, workerHistogram, N);
        printHistogram(histogram, N);
    }
    else
    {
        int **histograms, i, quarterSize, numOfThreads = omp_get_max_threads();
        int cudaHistogram[N] = { 0 };

        MPI_Recv(buff, BUFFER_SIZE, MPI_PACKED, ROOT, 0, MPI_COMM_WORLD, &status);
        MPI_Unpack(buff, BUFFER_SIZE, &position, &halfSize, 1, MPI_INT, MPI_COMM_WORLD);
        numbers = (int*)doMalloc(halfSize * sizeof(int));
        MPI_Unpack(buff, BUFFER_SIZE, &position, numbers, halfSize, MPI_INT, MPI_COMM_WORLD);

        quarterSize = halfSize / 2 + halfSize % 2;

        histograms = (int**)doMalloc(numOfThreads * sizeof(int*));
        for (i = 0; i < numOfThreads; i++)
        {
            histograms[i] = (int*)calloc(N, sizeof(int));
            if (!histograms[i])
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        
        histogramOpenMpPrivate(numbers, histograms, quarterSize);

        for (i = 0; i < numOfThreads; i++)
            mergeHistogram(workerHistogram, histograms[i], N);

        if (calculateHistogramCuda(numbers + quarterSize, cudaHistogram, quarterSize) != 0)
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        mergeHistogram(workerHistogram, cudaHistogram, N);

        MPI_Send(workerHistogram, N, MPI_INT, ROOT, 0, MPI_COMM_WORLD);

        for (i = 0; i < numOfThreads; i++)
            free(histograms[i]);
        free(histograms);
    }

    free(numbers);
    MPI_Finalize();
    return 0;
}