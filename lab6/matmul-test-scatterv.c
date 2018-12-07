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
	}

	int ndims = 2;
	int dims[2] = {3, 3};
	int periods[2] = {1, 1};
	int reorder = 0;
	MPI_Comm commCart;
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, commCart);

	int coords[2] = {0, 0};
	MPI_Cart_coords(commCart, rank, ndims, coords);

	MPI_Datatype block, blocktype;
	int displacments[9] = {0, 1, 2, 9, 10, 11, 18, 19, 20};
	int sendCount[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
	int recvCount = 9;
	MPI_Type_vector(3, 3, 9, MPI_INT, &block);
	MPI_Type_commit(&block); // not necessary
	MPI_Type_create_resized(block, 0, 3 * sizeof(int), &blocktype);
	MPI_Type_commit(&blocktype); // neededint
	MPI_Scatterv(a, sendCount, displacments, blocktype, mya, recvCount, MPI_INT, 0, commCart);

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
