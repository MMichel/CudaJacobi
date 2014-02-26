#include<stdio.h>
#include<stdlib.h>
#include<getopt.h>
#include <assert.h>
#include <cuda.h>
#include <time.h>

static char* program_name;

// Usage
void print_usage (FILE* stream, int exit_code)
{
  fprintf (stream, "Usage:  %s options\n", program_name);
  fprintf (stream,
           "  -h  --help             Display this usage information.\n"
           "  -f  --file filename    File containing coefficient matrix.\n"
           "  -i  --Ni int           Number of elements in Y direction (default=512).\n"
           "  -j  --Nj int           Number of elements in X direction (default=512).\n"
           "  -n  --iterations int   Number of iterations (default=10000).\n"
           "  -k  --kernel [1,2]     1: unoptimized, 2: optimized kernel (default).\n"
           "  -t  --tilesize int     Size of each thread block in kernel 2 (default=4).\n");
  exit (exit_code);
}


// Host version of the Jacobi method
void jacobiOnHost(float* x_next, float* A, float* x_now, float* b, int Ni, int Nj)
{
    int i,j;
    float sigma;

    for (i=0; i<Ni; i++)
    {
        sigma = 0.0;
        for (j=0; j<Nj; j++)
        {
            if (i != j)
                sigma += A[i*Nj + j] * x_now[j];
        }
        x_next[i] = (b[i] - sigma) / A[i*Nj + i];
    }
}


// Device version of the Jacobi method
__global__ void jacobiOnDevice(float* x_next, float* A, float* x_now, float* b, int Ni, int Nj)
{
    float sigma = 0.0;
    int idx = threadIdx.x;
    for (int j=0; j<Nj; j++)
    {
        if (idx != j)
            sigma += A[idx*Nj + j] * x_now[j];
    }
    x_next[idx] = (b[idx] - sigma) / A[idx*Nj + idx];
}


// Optimized device version of the Jacobi method
__global__ void jacobiOptimizedOnDevice(float* x_next, float* A, float* x_now, float* b, int Ni, int Nj)
{
    // Optimization step 1: tiling
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
     
    if (idx < Ni)
    {
        float sigma = 0.0;

        // Optimization step 2: store index in register
        // Multiplication is not executed in every iteration.
        int idx_Ai = idx*Nj;
        
        // Tried to use prefetching, but then the result is terribly wrong and I don't know why.. 
        /*     
        float curr_A = A[idx_Ai];
        float nxt_A;
        //printf("idx=%d\n",idx);
        for (int j=0; j<Nj-1; j++)
        {
            if (idx != j)
                nxt_A = A[idx_Ai + j + 1];
                sigma += curr_A * x_now[j];
                //sigma += A[idx_Ai + j] * x_now[j];
                curr_A = nxt_A;
                //printf("curr_A=%f\n",curr_A);
        }
        if (idx != Nj-1)
            sigma += nxt_A * x_now[Nj-1];
        x_next[idx] = (b[idx] - sigma) / A[idx_Ai + idx];
        */
        
        for (int j=0; j<Nj; j++)
            if (idx != j)
                sigma += A[idx_Ai + j] * x_now[j];

        // Tried to use loop-ennrolling, but also here this gives a wrong result.. 
        /*
        for (int j=0; j<Nj/4; j+=4)
        {
            if (idx != j)
            {
                sigma += A[idx_Ai + j] * x_now[j];
            }
            if (idx != j+1)
            {
                sigma += A[idx_Ai + j+1] * x_now[j+1];
            }
            if (idx != j+2)
            {
               sigma += A[idx_Ai + j+2] * x_now[j+2];
            }
            if (idx != j+3)
            {
                sigma += A[idx_Ai + j+3] * x_now[j+3];
            }
        }*/

        x_next[idx] = (b[idx] - sigma) / A[idx_Ai + idx];
    }
}


// device selection (copied from previous assignment)
static void selectGpu(int *gpu_num, int *num_devs)
{
    // gpu_num: (I/O): I: Default choice,
    //                 O: best device, changed only if more than one device
    // num_devs: (O)   Number of found devices.
    int best = *gpu_num;

    cudaGetDeviceCount(num_devs);
    if ( *num_devs > 1 )
    {
        int dev_num;
        int max_cores = 0;

        for (dev_num = 0; dev_num < *num_devs; dev_num++)
        {
            cudaDeviceProp dev_properties;

            cudaGetDeviceProperties(&dev_properties, dev_num);
            if (max_cores < dev_properties.multiProcessorCount)
            {
                max_cores = dev_properties.multiProcessorCount;
                best = dev_num;
            }
        }
        *gpu_num = best;
    }
}


// device test (copied from previous assignment)
static void testDevice(int devID)
{
    // Check if we can run. Maybe do something more...
    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, devID);
    if (deviceProp.major == 9999 && deviceProp.minor == 9999)
    {   /* Simulated device. */
        printf("There is no device supporting CUDA.\n");
        cudaThreadExit();
    }
    else
        printf("Using GPU device number %d.\n", devID);
}


int main(int argc, char *argv[])
{ 
    // initialize timing variables
    time_t start, end, start_h, end_h, start_d, end_d;
    float t_full, t_host, t_dev;

    start=clock();

    // initialize data variables
    float *x_now, *x_next, *A, *b, *x_h, *x_d;
    float *x_now_d, *x_next_d, *A_d, *b_d;

    // initialize parameter variables
    int N, Ni, Nj, iter, kernel, tileSize;    
    int ch;
    int i,k;
    char* fname;
    FILE* file;

    // Argument parsing
    static struct option long_options[] =
    {
        {"file", required_argument, NULL, 'f'},
        {"Ni", optional_argument, NULL, 'i'},
        {"Nj", optional_argument, NULL, 'j'},
        {"iterations", optional_argument, NULL, 'n'},
        {"kernel", optional_argument, NULL, 'k'},
        {"tilesize", optional_argument, NULL, 't'},
        {"help", optional_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}
    };

    program_name = argv[0];
    Ni=512, Nj=512, iter=10000, kernel=2, tileSize=4;
    ch=0;
    
    while ((ch = getopt_long(argc, argv,"f:i:j:n:k:h", long_options, NULL)) != -1) {
        switch (ch) {
             case 'f' : fname = optarg;
                 break;
             case 'i' : Ni = atoi(optarg);
                 break;
             case 'j' : Nj = atoi(optarg); 
                 break;
             case 'n' : iter = atoi(optarg);
                 break;
             case 'k' : kernel = atoi(optarg);
                 break;
             case 't' : tileSize = atoi(optarg);
                 break;
             case 'h': print_usage(stderr, 1); 
                 exit(EXIT_FAILURE);
             case '?': print_usage(stderr, 1); 
                 exit(EXIT_FAILURE);
             default: 
                 abort();
        }
    }

    N = Ni * Nj;


    printf("\nRunning Jacobi method:\n");
    printf("======================\n\n");
    printf("Coefficient matrix given in file: \n%s\n\n", fname);
    printf("Parameters:\n");
    printf("N=%d, Ni=%d, Nj=%d, ", N, Ni, Nj);
    printf("iterations=%d, kernel=%d, tilesize=%d\n", iter,kernel,tileSize);


    // Allocate memory on host
    x_next = (float *) malloc(Ni*sizeof(float));
    A = (float *) malloc(N*sizeof(float));
    x_now = (float *) malloc(Ni*sizeof(float));
    b = (float *) malloc(Ni*sizeof(float));
    x_h = (float *) malloc(Ni*sizeof(float));
    x_d = (float *) malloc(Ni*sizeof(float));

    // Initialize result vector x
    for (i=0; i<Ni; i++)
    {
        x_now[i] = 0;
        x_next[i] = 0;
    }

    // Read coefficient matrix from file
    file = fopen(fname, "r");
    if (file == NULL)
        exit(EXIT_FAILURE);
    char *line;
    size_t len = 0;
    i=0;
    while ((getline(&line, &len, file)) != -1) 
    {
        if (i<N)
            A[i] = atof(line);
        else
            b[i-N] = atof(line);
        i++;
    }
   

    start_h = clock();

    // Run "iter" iterations of the Jacobi method on HOST
    for (k=0; k<iter; k++)
    {
        if (k%2)
            jacobiOnHost(x_now, A, x_next, b, Ni, Nj);
        else
            jacobiOnHost(x_next, A, x_now, b, Ni, Nj);
        //for (i=0; i<Nj; i++)
        //    x_now[i] = x_next[i];
    }
    
    end_h = clock();

    // Save result from host in x_h
    for (i=0; i<Nj; i++)
        x_h[i] = x_next[i];


    // Re-initialize result vector x for device computation
    for (i=0; i<Ni; i++)
    {
        x_now[i] = 0;
        x_next[i] = 0;
    }


    // Check available device.
    int devID = 0, num_devs = 1;
    selectGpu(&devID, &num_devs);
    testDevice(devID);
  
    // Allocate memory on the device
    assert(cudaSuccess == cudaMalloc((void **) &x_next_d, Ni*sizeof(float)));
    assert(cudaSuccess == cudaMalloc((void **) &A_d, N*sizeof(float)));
    assert(cudaSuccess == cudaMalloc((void **) &x_now_d, Ni*sizeof(float)));
    assert(cudaSuccess == cudaMalloc((void **) &b_d, Ni*sizeof(float)));

    // Copy data -> device
    cudaMemcpy(x_next_d, x_next, sizeof(float)*Ni, cudaMemcpyHostToDevice);
    cudaMemcpy(A_d, A, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(x_now_d, x_now, sizeof(float)*Ni, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(float)*Ni, cudaMemcpyHostToDevice);

    // Compute grid and block size.
    // Un-optimized kernel
    int blockSize = Ni;
    int nBlocks = 1;

    // Optimized kernel
    int nTiles = Ni/tileSize + (Ni%tileSize == 0?0:1);
    int gridHeight = Nj/tileSize + (Nj%tileSize == 0?0:1);
    int gridWidth = Ni/tileSize + (Ni%tileSize == 0?0:1);
    printf("w=%d, h=%d\n",gridWidth,gridHeight);
    dim3 dGrid(gridHeight, gridWidth),
        dBlock(tileSize, tileSize);


    start_d = clock();
     
    // Run "iter" iterations of the Jacobi method on DEVICE
    if (kernel == 1)
    {
        printf("Using un-optimized kernel.\n");
        for (k=0; k<iter; k++)
        {
            if (k%2)
                jacobiOnDevice <<< nBlocks, blockSize >>> (x_now_d, A_d, x_next_d, b_d, Ni, Nj);
            else
                jacobiOnDevice <<< nBlocks, blockSize >>> (x_next_d, A_d, x_now_d, b_d, Ni, Nj);
            //cudaMemcpy(x_now_d, x_next_d, sizeof(float)*Ni, cudaMemcpyDeviceToDevice);
        }
    }
    else
    {
        printf("Using optimized kernel.\n");
        for (k=0; k<iter; k++)
        {
            if (k%2)
                jacobiOptimizedOnDevice <<< nTiles, tileSize >>> (x_now_d, A_d, x_next_d, b_d, Ni, Nj);
            else
                jacobiOptimizedOnDevice <<< nTiles, tileSize >>> (x_next_d, A_d, x_now_d, b_d, Ni, Nj);
            //cudaMemcpy(x_now_d, x_next_d, sizeof(float)*Ni, cudaMemcpyDeviceToDevice);
        }
    }
        
    end_d = clock();


    // Data <- device
    cudaMemcpy(x_d, x_next_d, sizeof(float)*Ni, cudaMemcpyDeviceToHost);
    
    // Free memory
    free(x_next); free(A); free(x_now); free(b);
    cudaFree(x_next_d); cudaFree(A_d); cudaFree(x_now_d); cudaFree(b_d);

    end=clock(); 

    printf("\nResult after %d iterations:\n",iter);
    float err = 0.0;
    for (i=0; i < Ni; i++)
    {
        //printf("x_h[%d]=%f\n",i,x_h[i]);
        //printf("x_d[%d]=%f\n",i,x_d[i]);
        err += abs(x_h[i] - x_d[i]) / Ni;
    }
    printf("x_h[%d]=%f\n",0,x_h[0]);
    printf("x_d[%d]=%f\n",0,x_d[0]);
    t_full = ((float)end - (float)start) / CLOCKS_PER_SEC;
    t_host = ((float)end_h - (float)start_h) / CLOCKS_PER_SEC;
    t_dev = ((float)end_d - (float)start_d) / CLOCKS_PER_SEC;
    printf("\nTiming:\nFull: %f\nHost: %f\nDevice: %f\n\n", t_full, t_host, t_dev);
    printf("Relative error: %f\n", err);

    printf("\nProgram terminated successfully.\n");
    return 0;
}
