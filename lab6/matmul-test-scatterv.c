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
#define P 9
#define SQRP 3

void main(int argc, char *argv[])
{
	FILE *f;
	int i, j, k, error, rank, size;
	float a[N][N], b[N][N], c[N][N], myc[SQRP][SQRP], mya[SQRP][SQRP], myb[SQRP][SQRP], mytmp[SQRP][SQRP];
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
		printf("RANK: %d (%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f) \n", rank, i, j, mya[i][0], myb[i][0], mya[i][1], myb[i][1], mya[i][2], myb[i][2]);
	}

	int aSendCords[2] = {(x - y + 3) % 3, y};
	int aSendRank;
	int aSendCount = N * N / P;
	MPI_Cart_rank(commCart, aSendCords, &aSendRank);
	MPI_Isend(mya, aSendCount, MPI_FLOAT, aSendRank, 0, commCart, &sendreq);

	int aRecvCords[2] = {(x + y) % 3, y};
	int aRecvRank;
	MPI_Cart_rank(commCart, aRecvCords, &aRecvRank);
	MPI_Irecv(mytmp, aSendCount, MPI_FLOAT, aRecvRank, 0, commCart, &rcvreq);

	MPI_Wait(&rcvreq, &status);

	MPI_BARRIER(commCart);

	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			mya[i][j] = mytmp[i][j];
		}
	}

	for (i = 0; i < SQRP; i++)
	{
		printf("RANK: %d (%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f) \n", rank, i, j, mya[i][0], myb[i][0], mya[i][1], myb[i][1], mya[i][2], myb[i][2]);
	}

	MPI_Finalize();
}