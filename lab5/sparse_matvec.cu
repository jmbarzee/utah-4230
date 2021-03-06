#include <stdio.h>

extern int cudaMemcpy();
extern int cudaFree();
extern void __syncthreads();
extern int cudaMemcpyToSymbol();
extern __global__ void computeGPU(int nr, int *ptr, int *indices, double *b, double *data, double *tgpu);

int compare(double *a, double *b, int size, double threshold)
{
  int i;
  for (i = 0; i < size; i++)
  {
    if (abs(a[i] - b[i]) > threshold)
      return 0;
  }
  return 1;
}

void computeCPU(int nr, int *ptr, int *indices, double *b, double *data, double *tcpu)
{
  int i, j;
  for (i = 0; i < nr; i++)
  {
    for (j = ptr[i]; j < ptr[i + 1]; j++)
    {
      tcpu[i] = tcpu[i] + data[j] * b[indices[j]];
    }
  }
}

extern __global__ void computeGPU(int nr, int *ptr, int *indices, double *b, double *data, double *tgpu)
{
  int i, j;
  //int bx = blockIdx.x;
  int tx = threadIdx.x;
  //double tgpucpy;
  __shared__ double bcpy[32];

  for (i = 0; i < nr; i++)
  {
    bcpy[tx] = b[blockDim.x * i + tx];
    __syncthreads(); // so that threads can operate on next section in lock step
    for (j = ptr[i]; j < ptr[i + 1]; j++)
    {
      tgpu[i] = tgpu[i] + data[j] * bcpy[indices[j]];
    }
    __syncthreads();
  }
}

main(int argc, char **argv)
{
  FILE *fp;
  char line[1024];
  int *ptr, *indices;
  double *data, *b, *tcpu, *tgpu;
  int i;
  int n;  // number of nonzero elements in data
  int nr; // number of rows in matrix
  int nc; // number of columns in matrix

  printf("Checking args\n");

  // Open input file and read to end of comments
  if (argc != 2)
  {
    printf("Not enough arguemnts!\n:");
  }

  if ((fp = fopen(argv[1], "r")) == NULL)
  {
    printf("File Failed to open!\n");
  }

  printf("Skipping comments\n");
  fgets(line, 128, fp);
  while (line[0] == '%')
  {
    fgets(line, 128, fp);
  }

  printf("Allocating sparse Array\n");
  // Read number of rows (nr), number of columns (nc) and
  // number of elements and allocate memory for ptr, indices, data, b and t.
  sscanf(line, "%d %d %d\n", &nr, &nc, &n);
  ptr = (int *)malloc((nr + 1) * sizeof(int));
  indices = (int *)malloc(n * sizeof(int));
  data = (double *)malloc(n * sizeof(double));
  b = (double *)malloc(nc * sizeof(double));
  tcpu = (double *)malloc(nr * sizeof(double));
  tgpu = (double *)malloc(nr * sizeof(double));

  printf("Reading Sparse Array\n");
  // Read data in coordinate format and initialize sparse matrix
  int lastr = 0;
  for (i = 0; i < n; i++)
  {
    int r;
    fscanf(fp, "%d %d %f\n", &r, &(indices[i]), &(data[i]));
    indices[i]--; // start numbering at 0
    if (r != lastr)
    {
      ptr[r - 1] = i;
      lastr = r;
    }
  }
  ptr[nr] = n;

  printf("Filling t with 0 and b with random numbers\n");
  // initialize t to 0 and b with random data
  for (i = 0; i < nr; i++)
  {
    tcpu[i] = 0.0;
    tgpu[i] = 0.0;
  }
  for (i = 0; i < nc; i++)
  {
    b[i] = (double)rand() / 1111111111;
  }

  printf("Setup Timing\n");
  // create CUDA event handles for timing purposes
  cudaEvent_t start_event, stop_event;
  float elapsed_time_cpu, elapsed_time_gpu;
  elapsed_time_cpu = 0;
  elapsed_time_gpu = 0;

  printf("Run CPU comp\n");
  // Main Computation, CPU version
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  cudaEventRecord(start_event, 0);
  computeCPU(nr, ptr, indices, b, data, tcpu);
  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  cudaEventElapsedTime(&elapsed_time_cpu, start_event, stop_event);


  printf("Setup GPU (copies)\n");
  // outputs
  double *devO1Ptr;
  cudaMalloc((void **)&devO1Ptr, nr * 4);

  // inputs
  int *devI1Ptr;
  cudaMalloc((void **)&devI1Ptr, (nr + 1) * 4);
  cudaMemcpy(devI1Ptr, ptr, (nr + 1) * 4, cudaMemcpyHostToDevice);
  int *devI2Ptr;
  cudaMalloc((void **)&devI2Ptr, n * 4);
  cudaMemcpy(devI2Ptr, indices, n * 4, cudaMemcpyHostToDevice);
  double *devI3Ptr;
  cudaMalloc((void **)&devI3Ptr, nc * 8);
  cudaMemcpy(devI3Ptr, b, nc * 8, cudaMemcpyHostToDevice); // this line is broken
  double *devI4Ptr;
  cudaMalloc((void **)&devI4Ptr, n * 8);
  cudaMemcpy(devI4Ptr, data, n * 8, cudaMemcpyHostToDevice);

  dim3 dimGrid((n + 31) / 32, 1);
  dim3 dimBlock(32, 1);

  printf("Run GPU comp\n");
  // Main Computation, GPU version
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  cudaEventRecord(start_event, 0);
  computeGPU<<<dimGrid, dimBlock>>>(nr, devI1Ptr, devI2Ptr, devI3Ptr, devI4Ptr, devO1Ptr);
  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  cudaMemcpy(tgpu, devO1Ptr, nr * 4, cudaMemcpyDeviceToHost);
  cudaEventElapsedTime(&elapsed_time_gpu, start_event, stop_event);

  printf("Comparing Results\n");
  // Compare computations to ensure correctness of gpu
  int res = compare(tcpu, tgpu, nr, 0.001);
  if (res == 1)
  {
    printf("VALID!\n  Sequential Time: %.2f msec\n  Parallel Time: %.2f msec\n Speedup = %.2f\n", elapsed_time_cpu, elapsed_time_gpu, elapsed_time_cpu / elapsed_time_gpu);
  }
  else
  {
    printf("INVALID...\n");
  }

  // TODO: Compute result on GPU and compare output
}