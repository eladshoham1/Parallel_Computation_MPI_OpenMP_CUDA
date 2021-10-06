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

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

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

        #pragma omp parallel for reduction(+: histogram)
            for (int i = 0; i < halfSize + size % 2; i++)
                histogram[numbers[i]]++;

        MPI_Recv(workerHistogram, N, MPI_INT, WORKER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        mergeHistogram(histogram, workerHistogram, N);
        printHistogram(histogram, N);
    }
    else
    {
        int **histograms, i, quarterSize, numOfThreads;
        int cudaHistogram[N] = { 0 };

        MPI_Recv(buff, BUFFER_SIZE, MPI_PACKED, ROOT, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Unpack(buff, BUFFER_SIZE, &position, &halfSize, 1, MPI_INT, MPI_COMM_WORLD);
        numbers = (int*)doMalloc(halfSize * sizeof(int));
        MPI_Unpack(buff, BUFFER_SIZE, &position, numbers, halfSize, MPI_INT, MPI_COMM_WORLD);

        quarterSize = halfSize / 2;

        #pragma omp parallel
        {
            #pragma omp single
                numOfThreads = omp_get_num_threads();
        }

        histograms = (int**)doMalloc(numOfThreads * sizeof(int*));
        for (i = 0; i < numOfThreads; i++)
        {
            histograms[i] = (int*)calloc(N, sizeof(int));
            if (!histograms[i])
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        #pragma omp parallel for
            for (int i = 0; i < quarterSize + halfSize % 2; i++)
                histograms[omp_get_thread_num()][numbers[i]]++;

        for (i = 0; i < numOfThreads; i++)
            mergeHistogram(workerHistogram, histograms[i], N);

        if (calculateHistogramCuda(numbers + quarterSize + halfSize % 2, cudaHistogram, quarterSize) != EXIT_SUCCESS)
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        mergeHistogram(workerHistogram, cudaHistogram, N);

        MPI_Send(workerHistogram, N, MPI_INT, ROOT, 0, MPI_COMM_WORLD);

        for (i = 0; i < numOfThreads; i++)
            free(histograms[i]);
        free(histograms);
    }

    free(numbers);
    MPI_Finalize();
    return EXIT_SUCCESS;
}