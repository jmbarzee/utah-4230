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
	for (i = 0; i < SQRP; i++)
	{
		printf("pre: %d(%d,%d)\t(%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f) \n", rank, x, y, mya[i][0], myb[i][0], mya[i][1], myb[i][1], mya[i][2], myb[i][2]);
	}

	// Initialize Send for initial skew of a
	int aSendCords[2] = {y, (x - y + 3) % 3};
	int aSendRank;
	int aSendCount = N * N / P;
	MPI_Cart_rank(commCart, aSendCords, &aSendRank);
	MPI_Isend(mya, aSendCount, MPI_FLOAT, aSendRank, 0, commCart, &sendreq);

	// Initialize Recv for initial skew of a
	int aRecvCords[2] = {y, (x + y) % 3};
	int aRecvRank;
	MPI_Cart_rank(commCart, aRecvCords, &aRecvRank);
	MPI_Irecv(mytmp, aSendCount, MPI_FLOAT, aRecvRank, 0, commCart, &rcvreq);
	MPI_Wait(&rcvreq, &status);

	// Copy mya from mytmp (receive buffer)
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			mya[i][j] = mytmp[i][j];
		}
	}

	// Initialize Send for initial skew of b
	int bSendCords[2] = {(y - x + 3) % 3, x};
	int bSendRank;
	int bSendCount = N * N / P;
	MPI_Cart_rank(commCart, bSendCords, &bSendRank);
	MPI_Isend(myb, bSendCount, MPI_FLOAT, bSendRank, 0, commCart, &sendreq);

	// Initialize Recv for initial skew of b
	int bRecvCords[2] = {(y + x) % 3, x};
	int bRecvRank;
	MPI_Cart_rank(commCart, bRecvCords, &bRecvRank);
	MPI_Irecv(mytmp, bSendCount, MPI_FLOAT, bRecvRank, 0, commCart, &rcvreq);
	MPI_Wait(&rcvreq, &status);

	// Copy myb from mytmp (receive buffer)
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			myb[i][j] = mytmp[i][j];
		}
	}

	for (i = 0; i < SQRP; i++)
	{
		printf("post: %d(%d,%d)\t(%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f) \n", rank, x, y, mya[i][0], myb[i][0], mya[i][1], myb[i][1], mya[i][2], myb[i][2]);
	}

	MPI_Finalize();
}
