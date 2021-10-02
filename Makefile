build:
	mpicxx -fopenmp -c main.c
	mpicxx -fopenmp -c cFunctions.c
	nvcc -I./inc -c cudaFunctions.cu
	mpicxx -fopenmp -o histogram main.o cFunctions.o cudaFunctions.o /usr/local/cuda-9.1/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o ./histogram

run:
	mpiexec -n 2 ./histogram < input.txt