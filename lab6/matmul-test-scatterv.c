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
	int a[N][N], b[N][N], c[N][N], myc[SQRP][SQRP], mya[SQRP][SQRP], myb[SQRP][SQRP];
	MPI_Request sendreq, rcvreq;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// sequential MM
	if (rank == 0)
	{
		// initialize a and b
		int val = 0;
		for (i = 0; i < N; i++)
		{
			for (j = 0; j < N; j++)
			{
				a[i][j] = val++;
				b[i][j] = val++;
			}
		}

		// int ndims = 2;
		// int sizes[2] = {N, P};
		// int subSizes[2] = {SQRP, SQRP};
		// int starts[2] = {0, 0};

		// MPI_Datatype subArrayA;
		// MPI_Type_create_subarray(ndims, sizes, subSizes, starts, MPI_ORDER_C, MPI_INT, &subArrayA);
		// MPI_Type_commit(&subArrayA);
		// // MPI_Scatterv( sendbuf, scounts, displs, MPI_INT, rbuf, 100, MPI_INT, root, comm);

		// MPI_Datatype subArrayB;
		// MPI_Type_create_subarray(ndims, sizes, subSizes, starts, MPI_ORDER_C, MPI_INT, &subArrayB);
		// MPI_Type_commit(&subArrayB);
	}
	int rowRank, rowSize;
	MPI_Comm rowComm;
	MPI_Comm_split(MPI_COMM_WORLD, rank / (N / SQRP), rank, &rowComm);
	MPI_Comm_rank(rowComm, &rowRank);
	MPI_Comm_size(rowComm, &rowSize);

	int colRank, colSize;
	MPI_Comm colComm;
	MPI_Comm_split(MPI_COMM_WORLD, rank % (N / SQRP), rank, &colComm);
	MPI_Comm_rank(colComm, &colRank);
	MPI_Comm_size(colComm, &colSize);

	printf("WORLD: %d/%d \t ROW: %d/%d \t COL: %d/%d\n", rank, size, rowRank, rowSize, colRank, colSize);

	// TODO: Scatter 3x3 tiles to mya and myb

	for (i = 0; i < SQRP; i++)
	{
		for (j = 0; j < SQRP; j++)
		{
			printf("RANK: %d, a[%d][%d] = %d, b = %d\n", rank, i, j, mya[i][j], myb[i][j]);
		}
	}
	MPI_Finalize();
}
