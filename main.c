#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "functions.h"
#include "myProto.h"

int main(int argc, char *argv[])
{
    int *numbers, counters[N] = { 0 }, workerCounters[N] = { 0 };
    int size, halfSize, rank, position = 0;
    char buff[BUFFER_SIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == ROOT)
    {
        numbers = readNumbers(&size);
        halfSize = size / 2;

        MPI_Pack(&halfSize, 1 , MPI_INT, buff, BUFFER_SIZE, &position, MPI_COMM_WORLD);
        MPI_Pack(numbers + halfSize, halfSize, MPI_INT, buff, BUFFER_SIZE, &position, MPI_COMM_WORLD);
        MPI_Send(buff, position, MPI_PACKED, WORKER ,0 ,MPI_COMM_WORLD);

        calculateHistogramOpenMp(numbers, counters, halfSize + size % 2);

        MPI_Recv(workerCounters, halfSize, MPI_INT, WORKER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        mergeHistogram(counters, workerCounters, N);
        printHistogram(counters, N);
    }
    else
    {
        int **histograms, quarterSize, i, tid;
        int numOfThreads = omp_get_max_threads();
        printf("num of threads %d\n\n\n", numOfThreads);
        MPI_Recv(buff, BUFFER_SIZE, MPI_PACKED, ROOT, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Unpack(buff, BUFFER_SIZE, &position, &halfSize, 1, MPI_INT, MPI_COMM_WORLD);
        numbers = (int*)doMalloc(halfSize * sizeof(int));
        MPI_Unpack(buff, BUFFER_SIZE, &position, numbers, halfSize, MPI_INT, MPI_COMM_WORLD);

        quarterSize = halfSize / 2;

        histograms = (int**)doMalloc(numOfThreads * sizeof(int*));
        for (i = 0; i < numOfThreads; i++)
        {
            histograms[i] = (int*)calloc(N, sizeof(int));
            if (!histograms[i])
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

    #pragma omp parallel private (i, tid)
    {
        tid = omp_get_thread_num();
    #pragma omp for
        for (i=0; i<50; i++)
            printf("tid = %d\n", tid);
    }

    //#pragma omp parallel for
        for (i = 0; i < quarterSize + halfSize % 2; i++)
        {
            //int threadNum = omp_get_thread_num();
            //printf("thread num %d\n", threadNum);
            //calculateHistogramOpenMp(numbers, histograms[threadNum], halfSize);
            //mergeHistogram(workerCounters, histograms[threadNum], N);
        }

        /*int tempCounters[N] = { 0 };
        if (computeOnGPU(numbers + quarterSize, tempCounters, quarterSize) != 0)
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        mergeHistogram(workerCounters, tempCounters, N);*/

        MPI_Send(workerCounters, halfSize, MPI_INT, ROOT, 0, MPI_COMM_WORLD);

        for (i = 0; i < numOfThreads; i++)
            free(histograms[i]);
        free(histograms);
    }

    free(numbers);
    MPI_Finalize();
    return 0;
}