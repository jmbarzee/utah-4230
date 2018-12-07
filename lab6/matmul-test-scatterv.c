// Compiling:
//    module load intel
//    mpicc matmul-test-scatterv.c -o matmul
// Executing:
//    mpiexec -n 9 matmul
// Sbatch execution:
//    sbatch script.matmul

#include "stdio.h"
#include "mpi.h"

#define N 9
#define SQRP 3

void main(int argc, char *argv[])
{
	FILE *f;
	int i, j, k, error, rank, size;
	float a[N][N], b[N][N], c[N][N], myc[SQRP][SQRP], mya[SQRP][SQRP], myb[SQRP][SQRP];
	MPI_Request sendreq, rcvreq;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// sequential MM
	if (rank == 0)
	{
		// initialize a and b
		float val = 0;
		for (i = 0; i < N; i++)
		{
			for (j = 0; j < N; j++)
			{
				a[i][j] = val++;
				b[i][j] = val++;
			}
		}
	}

	int ndims = 2;
	int dims[2] = {3, 3};
	int periods[2] = {1, 1};
	int reorder = 0;
	MPI_Comm commCart;
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &commCart);

	int coords[2] = {0, 0};
	MPI_Cart_coords(commCart, rank, ndims, coords);
	int x = coords[0];
	int y = coords[1];

	MPI_Datatype block, blocktype;
	int displacments[9] = {0, 1, 2, 9, 10, 11, 18, 19, 20};
	int sendCount[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
	int recvCount = 9;
	MPI_Type_vector(3, 3, 9, MPI_FLOAT, &block);
	MPI_Type_commit(&block); // not necessary
	MPI_Type_create_resized(block, 0, 3 * sizeof(float), &blocktype);
	MPI_Type_commit(&blocktype); // neededint
	MPI_Scatterv(a, sendCount, displacments, blocktype, mya, recvCount, MPI_FLOAT, 0, commCart);
	MPI_Scatterv(b, sendCount, displacments, blocktype, myb, recvCount, MPI_FLOAT, 0, commCart);

	// TODO: Scatter 3x3 tiles to mya and myb

	for (i = 0; i < SQRP; i++)
	{
		printf("RANK: %d", rank);
		for (j = 0; j < SQRP; j++)
		{
			printf("\t(%.2f, %.2f)", mya[i][j], myb[i][j]);
		}
		printf("\n");
	}
	printf("\n\n");

	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			if (i/3 == y && j/3 == x) {
				printf("\t(%3.0f, %3.0f)", mya[i%3][j%3], myb[i%3][j%3]);
			}
		}
		printf("\n");
	}
	printf("\n\n");

	// MPI_Isend(mya, N * N / P, MPI_FLOAT, (rank + 1) % P, 0, MPI_COMM_WORLD, &sendreq);
	// int src = (rank == 0) ? P - 1 : rank - 1;
	// MPI_Irecv(mytmp, N * N / P, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &rcvreq);
	// MPI_Wait(&rcvreq, &status);

	MPI_Finalize();
}
