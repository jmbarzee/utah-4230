#include <stdio.h>


#define __min(x,y) ((x)<(y)?(x):(y))

;
#define N 1024

extern int cudaMemcpy();
extern int cudaFree();
extern void __syncthreads();
extern int cudaMemcpyToSymbol();
extern __global__ void mv_GPU(float *, float (*)[N], float *);

int compare(float *a, float *b, int size, double threshold) {
    int i;
    for (i=0; i<size; i++) {
      if (abs(a[i]-b[i]) > threshold) return 0;
    }
    return 1;
}

void normalMV(float *a, float *c, float *b){
  int i;
  int j;
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
		a[i] = a[i] +  c[j*N+i] * b[j] ;
}

extern __global__ void mv_GPU(float *a, float (*c)[N], float *b)
  {
    int bx;
    int tx;
    float acpy;
    __shared__ float bcpy[32];
    int k;
    int j;

    bx = blockIdx.x;
    tx = threadIdx.x;
    int i = bx * blockDim.x + tx;
    int tiles = (N+blockDim.x-1)/blockDim.x;

    if (tx <= -(blockDim.x * bx) + N-1)
      {
        acpy = 0.0;
//        suif_tmp0 = ((float *)(float (*)[])a)[i];
      
    for (k = 0; k <= tiles; k++)
      {
          bcpy[tx] = b[blockDim.x * k + tx];
          __syncthreads();
          for (j = 32 * k; j <= __min(32 * k + 31, N-1); j++)
          {
              acpy = acpy + c[j][i] * bcpy[j - 32 * k];
          }
        __syncthreads();
      }
        a[i] = acpy;
      }
  }

main (int argc, char **argv) {
  float *c, h_a[N], d_a[N], b[N];  

  // allocate memory for c, and initialize a, b and c
  c = (float *) malloc((N)*(N)*sizeof(float));
  for( int j=0;j<N;j++) {
      b[j] = (float) rand()/RAND_MAX;
  }
  for(int i=0;i<N;i++) {
      for( int j=0;j<N;j++) {
         c[i*N+j] = (float) rand()/RAND_MAX;
      }
  }

  // initialize host output to 0
  for( int j=0;j<N;j++) 
     h_a[j] = 0.0;
          

  // create CUDA event handles for timing purposes
  cudaEvent_t start_event, stop_event;
  float elapsed_time_seq, elapsed_time_gpu;

  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  cudaEventRecord(start_event, 0);   
  normalMV(h_a, c, b);
  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  cudaEventElapsedTime(&elapsed_time_seq,start_event, stop_event);

    float *devO1Ptr;
    float *devI1Ptr;
    float *devI2Ptr;

    cudaMalloc((void **)&devO1Ptr, N * 4);
    cudaMalloc((void **)&devI1Ptr, N*N * 4);
    cudaMemcpy(devI1Ptr, c, N*N * 4, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&devI2Ptr, N * 4);
    cudaMemcpy(devI2Ptr, b, N * 4, cudaMemcpyHostToDevice);

    dim3 dimGrid((N+31)/32, 1);
    dim3 dimBlock(32, 1);

  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  cudaEventRecord(start_event, 0);   
  mv_GPU<<<dimGrid,dimBlock>>>(devO1Ptr, (float(*) [N])devI1Ptr, devI2Ptr);
  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  cudaMemcpy(d_a, devO1Ptr, N * 4, cudaMemcpyDeviceToHost);
  cudaEventElapsedTime(&elapsed_time_gpu,start_event, stop_event);


    cudaFree(devO1Ptr);
    cudaFree(devI1Ptr);
    cudaFree(devI2Ptr);


  int res = compare( h_a, d_a, N, 0.001);
  if (res == 1) {
    printf("VALID!\n  Sequential Time: %.2f msec\n  Parallel Time: %.2f msec\n Speedup = %.2f\n", elapsed_time_seq, elapsed_time_gpu, elapsed_time_seq/elapsed_time_gpu);
  }
  else printf("INVALID...\n");
}
