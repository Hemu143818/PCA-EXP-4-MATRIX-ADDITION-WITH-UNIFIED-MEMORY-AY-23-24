# PCA-EXP-4-MATRIX-ADDITION-WITH-UNIFIED-MEMORY AY 23-24
<h3>AIM:</h3>
<h3>ENTER YOUR NAME :k hemanth yadav</h3>
<h3>ENTER YOUR REGISTER NO :212224100033 </h3> 
<h3>EX. NO :ex:-4 </h3>
<h3>DATE 25-03-2026 </h3>  

<h1> <align=center> MATRIX ADDITION WITH UNIFIED MEMORY </h3>
  Refer to the program sumMatrixGPUManaged.cu. Would removing the memsets below affect performance? If you can, check performance with nvprof or nvvp.</h3>

## AIM:
To perform Matrix addition with unified memory and check its performance with nvprof.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
## PROCEDURE:
1.	Setup Device and Properties
Initialize the CUDA device and get device properties.
2.	Set Matrix Size: Define the size of the matrix based on the command-line argument or default value.
Allocate Host Memory
3.	Allocate memory on the host for matrices A, B, hostRef, and gpuRef using cudaMallocManaged.
4.	Initialize Data on Host
5.	Generate random floating-point data for matrices A and B using the initialData function.
6.	Measure the time taken for initialization.
7.	Compute Matrix Sum on Host: Compute the matrix sum on the host using sumMatrixOnHost.
8.	Measure the time taken for matrix addition on the host.
9.	Invoke Kernel
10.	Define grid and block dimensions for the CUDA kernel launch.
11.	Warm-up the kernel with a dummy launch for unified memory page migration.
12.	Measure GPU Execution Time
13.	Launch the CUDA kernel to compute the matrix sum on the GPU.
14.	Measure the execution time on the GPU using cudaDeviceSynchronize and timing functions.
15.	Check for Kernel Errors
16.	Check for any errors that occurred during the kernel launch.
17.	Verify Results
18.	Compare the results obtained from the GPU computation with the results from the host to ensure correctness.
19.	Free Allocated Memory
20.	Free memory allocated on the device using cudaFree.
21.	Reset Device and Exit
22.	Reset the device using cudaDeviceReset and return from the main function.

## PROGRAM:
```
%%cuda
#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

inline double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1e-6;
}

__global__ void sumMatrixGPU(float *A, float *B, float *C, int nx, int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix < nx && iy < ny)
    {
        int idx = iy * nx + ix;
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    printf("Unified Memory Matrix Addition with Warm-up\n");

    int nx = 1 << 12;
    int ny = 1 << 12;
    int n = nx * ny;
    size_t size = n * sizeof(float);

    float *A, *B, *C;

    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    // Initialize data
    for (int i = 0; i < n; i++)
    {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    dim3 block(32, 32);
    dim3 grid((nx + 31) / 32, (ny + 31) / 32);

    sumMatrixGPU<<<grid, block>>>(A, B, C, nx, ny);
    cudaDeviceSynchronize();

    double start = seconds();

    sumMatrixGPU<<<grid, block>>>(A, B, C, nx, ny);
    cudaDeviceSynchronize();

    double time = seconds() - start;

    printf("Optimized Execution Time: %f sec\n", time);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
```

## OUTPUT:
<img width="617" height="84" alt="image" src="https://github.com/user-attachments/assets/2acd4cc1-6022-407c-8610-fec280e3af2a" />


## RESULT:
Thus the program has been successfully executed using unified memory for matrix addition. It is observed that removing the memset() function gives less elapsed time (0.000837 sec)
