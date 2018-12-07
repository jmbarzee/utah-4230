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
				a[i][j] = val;
				b[i][j] = val;
				val++;
			}
		}
	}

	// Initialize catersian communicator
	int ndims = 2;
	int dims[2] = {3, 3};
	int periods[2] = {1, 1};
	int reorder = 0;
	MPI_Comm commCart;
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &commCart);

	// Grab coordinates of processor
	int coords[2] = {0, 0};
	MPI_Cart_coords(commCart, rank, ndims, coords);
	int x = coords[1];
	int y = coords[0];

	// Scatter a and b / Initialize mya and myb
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

	// Dump
	printf("pre: %d(%d,%d)\n\
		(%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f)\n\
		(%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f)\n\
		(%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f)\n\n",
		   rank, x, y,
		   mya[0][0], myb[0][0], mya[0][1], myb[0][1], mya[0][2], myb[0][2],
		   mya[1][0], myb[1][0], mya[1][1], myb[1][1], mya[1][2], myb[1][2],
		   mya[2][0], myb[2][0], mya[2][1], myb[2][1], mya[2][2], myb[2][2]);

	// // Shift a
	// int aDisplacment = -1;
	// int aDirection = 1; // x dimension
	// int aRankSource, aRankDest;
	// int aSendCount = N * N / P;
	// MPI_Cart_shift(commCart, aDirection, aDisplacment, &aRankSource, &aRankDest);
	// MPI_Isend(mya, aSendCount, MPI_FLOAT, aRankDest, 0, commCart, &sendreq);
	// MPI_Irecv(mytmp, aSendCount, MPI_FLOAT, aRankSource, 0, commCart, &rcvreq);
	// MPI_Wait(&rcvreq, &status);

	// // Copy mya from mytmp (receive buffer)
	// for (i = 0; i < SQRP; i++)
	// {
	// 	for (j = 0; j < SQRP; j++)
	// 	{
	// 		mya[i][j] = mytmp[i][j];
	// 	}
	// }

	// // Dump
	// printf("postA: %d(%d,%d)\n\
	// 	(%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f)\n\
	// 	(%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f)\n\
	// 	(%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f)\n\n",
	// 	   rank, x, y,
	// 	   mya[0][0], myb[0][0], mya[0][1], myb[0][1], mya[0][2], myb[0][2],
	// 	   mya[1][0], myb[1][0], mya[1][1], myb[1][1], mya[1][2], myb[1][2],
	// 	   mya[2][0], myb[2][0], mya[2][1], myb[2][1], mya[2][2], myb[2][2]);
	// 	   	// Shift a

	int bDisplacment = -1;
	int bDirection = 0; // x dimension
	int bRankSource, bRankDest;
	int bSendCount = N * N / P;
	MPI_Cart_shift(commCart, bDirection, bDisplacment, &bRankSource, &bRankDest);
	MPI_Isend(myb, bSendCount, MPI_FLOAT, bRankDest, 0, commCart, &sendreq);
	MPI_Irecv(mytmp, bSendCount, MPI_FLOAT, bRankSource, 0, commCart, &rcvreq);
	MPI_Wait(&rcvreq, &status);

	// Copy mya from mytmp (receive buffer)
	for (i = 0; i < SQRP; i++)
	{
		for (j = 0; j < SQRP; j++)
		{
			myb[i][j] = mytmp[i][j];
		}
	}

	// Dump
	printf("postB: %d(%d,%d)\n\
		(%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f)\n\
		(%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f)\n\
		(%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f)\n\n",
		   rank, x, y,
		   mya[0][0], myb[0][0], mya[0][1], myb[0][1], mya[0][2], myb[0][2],
		   mya[1][0], myb[1][0], mya[1][1], myb[1][1], mya[1][2], myb[1][2],
		   mya[2][0], myb[2][0], mya[2][1], myb[2][1], mya[2][2], myb[2][2]);

	MPI_Finalize();
}
