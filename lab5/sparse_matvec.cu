#include <stdio.h>

extern int cudaMemcpy();
extern int cudaFree();
extern void __syncthreads();
extern int cudaMemcpyToSymbol();
void computeGPU(int nr, int *ptr, int *indices, float *b, float *data, float *tgpu);

int compare(float *a, float *b, int size, double threshold)
{
  int i;
  for (i = 0; i < size; i++)
  {
    if (abs(a[i] - b[i]) > threshold)
      return 0;
  }
  return 1;
}

void computeCPU(int nr, int *ptr, int *indices, float *b, float *data, float *tcpu)
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

void computeGPU(int nr, int *ptr, int *indices, float *b, float *data, float *tgpu)
{
  int i, j;
  for (i = 0; i < nr; i++)
  {
    for (j = ptr[i]; j < ptr[i + 1]; j++)
    {
      tgpu[i] = tgpu[i] + data[j] * b[indices[j]];
    }
  }
}

main(int argc, char **argv)
{
  FILE *fp;
  char line[1024];
  int *ptr, *indices;
  float *data, *b, *tcpu, *tgpu;
  int i;
  int n;  // number of nonzero elements in data
  int nr; // number of rows in matrix
  int nc; // number of columns in matrix

  // Open input file and read to end of comments
  if (argc != 2)
  {
    abort();
  }

  if ((fp = fopen(argv[1], "r")) == NULL)
  {
    abort();
  }

  fgets(line, 128, fp);
  while (line[0] == '%')
  {
    fgets(line, 128, fp);
  }

  // Read number of rows (nr), number of columns (nc) and
  // number of elements and allocate memory for ptr, indices, data, b and t.
  sscanf(line, "%d %d %d\n", &nr, &nc, &n);
  ptr = (int *)malloc((nr + 1) * sizeof(int));
  indices = (int *)malloc(n * sizeof(int));
  data = (float *)malloc(n * sizeof(float));
  b = (float *)malloc(nc * sizeof(float));
  tcpu = (float *)malloc(nr * sizeof(float));
  tgpu = (float *)malloc(nr * sizeof(float));

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

  // initialize t to 0 and b with random data
  for (i = 0; i < nr; i++)
  {
    tcpu[i] = 0.0;
    tgpu[i] = 0.0;
  }
  for (i = 0; i < nc; i++)
  {
    b[i] = (float)rand() / 1111111111;
  }

  // create CUDA event handles for timing purposes
  cudaEvent_t start_event, stop_event;
  float elapsed_time_cpu, elapsed_time_gpu;
  elapsed_time_cpu = 0;
  elapsed_time_gpu = 0;

  // Main Computation, CPU version
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  cudaEventRecord(start_event, 0);
  computeCPU(nr, ptr, indices, b, data, tcpu);
  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  cudaEventElapsedTime(&elapsed_time_cpu, start_event, stop_event);

  // Main Computation, GPU version
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
  cudaEventRecord(start_event, 0);
  computeGPU(nr, ptr, indices, b, data, tgpu);
  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  cudaEventElapsedTime(&elapsed_time_gpu, start_event, stop_event);

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