CC=nvcc
CFLAGS=-I/usr/local/cuda-5.5/include -I/cfs/zorn/nobackup/m/mircom/cuda/samples/common/inc
LDFLAGS=-L${CUDA_HOME}/lib64 

all: jacobi

jacobi: jacobi.cu
	$(CC) -o jacobi $(CFLAGS) -arch sm_20 $(LDFLAGS) jacobi.cu
	
clean: 
	rm jacobi 
