/*
 * This file was created automatically from SUIF
 *   on Fri Nov  4 15:08:58 2011.
 */
#include <stdio.h>
//#include <cutil.h>

#define __suif_min(x,y) ((x)<(y)?(x):(y))

;
#define N 4096
extern void MV_GPU_wrapper(float (*)[4096], float *, float *);

extern int cudaMemcpy();
extern int cudaFree();
extern void __syncthreads();
extern int cudaMemcpyToSymbol();
extern __global__ void mv_GPU(float *, float (*)[4096], float *);

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

extern void MV_GPU_wrapper(float *a, float *c, float *b)
  {
    return;
  }

extern __global__ void mv_GPU(float *a, float (*c)[4096], float *b)
  {
    int bx;
    int tx;
    float suif_tmp0;
    __shared__ float _P1[96];
    int k;
    int j;

    bx = blockIdx.x;
    tx = threadIdx.x;
    if (tx <= -(96 * bx) + 4095)
      {
        suif_tmp0 = 0.0;
//        suif_tmp0 = ((float *)(float (*)[])a)[tx + 96 * bx];
      }
    for (k = 0; k <= 42; k++)
      {
        if (tx <= -(96 * k) + 4095)
          {
            ((float *)(float (*)[96])_P1)[96 * k + tx - 96 * k] = ((float *)(float (*)[])b)[96 * k + tx];
          }
        __syncthreads();
        if (tx <= -(96 * bx) + 4095)
          {
            for (j = 96 * k; j <= __suif_min(96 * k + 95, 4095); j++)
              {
                suif_tmp0 = suif_tmp0 + ((float (*)[4096])(float ((*)[])[4096])c)[j][96 * bx + tx] * ((float *)(float (*)[96])_P1)[j - 96 * k];
              }
          }
        __syncthreads();
      }
    if (tx <= -(96 * bx) + 4095)
      {
        ((float *)(float (*)[])a)[tx + 96 * bx] = suif_tmp0;
      }
  }

main (int argc, char **argv) {
  float *c, h_a[4096], d_a[4096], b[4096];  

  // allocate memory for c, and initialize a, b and c
  c = (float *) malloc((4096)*(4096)*sizeof(float));
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

    cudaMalloc((void **)&devO1Ptr, 4096 * 4);
    cudaMalloc((void **)&devI1Ptr, 16777216 * 4);
    cudaMemcpy(devI1Ptr, c, 16777216 * 4, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&devI2Ptr, 4096 * 4);
    cudaMemcpy(devI2Ptr, b, 4096 * 4, cudaMemcpyHostToDevice);

    dim3 dimGrid(43, 1);
    dim3 dimBlock(96, 1);

  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  cudaEventRecord(start_event, 0);   
  mv_GPU<<<dimGrid,dimBlock>>>(devO1Ptr, (float(*) [4096])devI1Ptr, devI2Ptr);
  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  cudaMemcpy(d_a, devO1Ptr, 4096 * 4, cudaMemcpyDeviceToHost);
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
