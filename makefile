objects = main.o functions.o cudaFunctions.o
cudaPath = /usr/local/cuda-9.1/lib64/libcudart_static.a
flags = -fopenmp -ldl -lrt 

myprog: $(objects)
	mpicxx -o myprog $(objects) $(cudaPath) $(flags)

main.o: main.c
	mpicxx -fopenmp -c main.c

functions.o: functions.c
	mpicxx -fopenmp -c functions.c

cudaFunctions.o: cudaFunctions.cu
	nvcc -c cudaFunctions.cu

clean:
	rm -f myprog $(objects)

run:
	mpiexec -n 2 ./myprog < input.txt