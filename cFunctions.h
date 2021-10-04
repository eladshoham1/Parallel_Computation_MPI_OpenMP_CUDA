#ifndef __FUNCTIONS_H__
#define __FUNCTIONS_H__

enum ranks { ROOT, WORKER };
enum buffer { BUFFER_SIZE = 1024 * 1024 };
    
int* readNumbers(int* size);
void printHistogram(int* counters, int size);
void mergeHistogram(int* masterHistogram, int* workerHistogram, int size);
void* doMalloc(unsigned int nbytes);

#endif