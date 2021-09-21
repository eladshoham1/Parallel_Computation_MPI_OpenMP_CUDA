#ifndef __FUNCTIONS_H__
#define __FUNCTIONS_H__

enum ranks { ROOT, WORKER };
enum size { N = 256, BUFFER_SIZE = 1024 };
    
int* readNumbers(int* size);
void printHistogram(int *counters, int size);
void calculateHistogramOpenMp(int* numbers, int* counters, int size);
void mergeHistogram(int* masterCounters, int* workerCounters, int size);
void* doMalloc(unsigned int nbytes);

#endif