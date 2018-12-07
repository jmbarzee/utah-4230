// Compiling:
//    module load intel
//    mpicc matmul-assign.c -o matmul
// Executing:
//    mpiexec -n 2 matmul
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
	float a[N][N], b[N][N], c[N][N], myc[SQRP][SQRP], mya[SQRP][SQRP], myb[SQRP][SQRP], mytmp[SQRP][SQRP], tmpdata;
	MPI_Request sendreq, rcvreq;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// sequential MM
	if (rank == 0)
	{

		// read in matrix
		// f = fopen("matrixA.dat", "r");
		// for (i = 0; i < N; i++)
		// {
		// 	for (j = 0; j < N; j++)
		// 	{
		// 		error = fscanf(f, "%f", &tmpdata);
		// 		a[i][j] = tmpdata;
		// 		b[i][j] = tmpdata + 1.;
		// 	}
		// }
		// fclose(f);
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

		// After computing each point, output sequential results.
		for (i = 0; i < N; i++)
		{
			for (j = 0; j < N; j++)
			{
				c[i][j] = 0.;
				for (k = 0; k < N; k++)
				{
					c[i][j] += a[i][k] * b[k][j];
				}
				// printf("SEQ: c[%d][%d] = %f\n", i, j, c[i][j]);
			}
		}
	}

	// Initilize catersian communicator
	int ndims = 2;
	int dims[2] = {3, 3};
	int periods[2] = {1, 1};
	int reorder = 0;
	MPI_Comm commCart;
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &commCart);

	// Grab coordinates of processor
	int coords[2] = {0, 0};
	MPI_Cart_coords(commCart, rank, ndims, coords);
	int x = coords[0];
	int y = coords[1];

	// Scatter a and b / Initilize mya and myb
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

	// Initilize myc
	for (i = 0; i < SQRP; i++)
	{
		for (j = 0; j < SQRP; j++)
		{
			myc[i][j] = 0.0;
		}
	}
	// Dump
	printf("PostScatter: %d(%d,%d)\n\
		(%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f)\n\
		(%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f)\n\
		(%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f)\n\n",
		   rank, x, y,
		   mya[0][0], myb[0][0], mya[0][1], myb[0][1], mya[0][2], myb[0][2],
		   mya[1][0], myb[1][0], mya[1][1], myb[1][1], mya[1][2], myb[1][2],
		   mya[2][0], myb[2][0], mya[2][1], myb[2][1], mya[2][2], myb[2][2]);

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
	for (i = 0; i < SQRP; i++)
	{
		for (j = 0; j < SQRP; j++)
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

	// Dump
	printf("PostSqew: %d(%d,%d)\n\
		(%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f)\n\
		(%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f)\n\
		(%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f)\n\n",
		   rank, x, y,
		   mya[0][0], myb[0][0], mya[0][1], myb[0][1], mya[0][2], myb[0][2],
		   mya[1][0], myb[1][0], mya[1][1], myb[1][1], mya[1][2], myb[1][2],
		   mya[2][0], myb[2][0], mya[2][1], myb[2][1], mya[2][2], myb[2][2]);

	// Copy myb from mytmp (receive buffer)
	for (i = 0; i < SQRP; i++)
	{
		for (j = 0; j < SQRP; j++)
		{
			myb[i][j] = mytmp[i][j];
		}
	}

	// TODO: Add following loop:
	int q;
	for (q = 0; q < SQRP; q++)
	{
		for (i = 0; i < SQRP; i++)
		{
			for (j = 0; j < SQRP; j++)
			{
				for (k = 0; k < SQRP; k++)
				{
					myc[i][j] += mya[i][k] * myb[k][j];
				}
			}
		}

		// Shift a
		int aDisplacment = -1;
		int aDirection = 1; // x dimension
		int aRankSource, aRankDest;
		MPI_Cart_shift(commCart, aDirection, aDisplacment, &aRankSource, &aRankDest);
		MPI_Isend(mya, aSendCount, MPI_FLOAT, aRankDest, 0, commCart, &sendreq);
		MPI_Irecv(mytmp, aSendCount, MPI_FLOAT, aRankSource, 0, commCart, &rcvreq);
		MPI_Wait(&rcvreq, &status);

		// Copy mya from mytmp (receive buffer)
		for (i = 0; i < SQRP; i++)
		{
			for (j = 0; j < SQRP; j++)
			{
				mya[i][j] = mytmp[i][j];
			}
		}

		// Shift b
		int bDisplacment = -1;
		int bDirection = 0; // x dimension
		int bRankSource, bRankDest;
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
		printf("PostShift%d: %d(%d,%d)\n\
		(%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f)\n\
		(%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f)\n\
		(%3.0f, %3.0f) (%3.0f, %3.0f) (%3.0f, %3.0f)\n\n",
			   q, rank, x, y,
			   mya[0][0], myb[0][0], mya[0][1], myb[0][1], mya[0][2], myb[0][2],
			   mya[1][0], myb[1][0], mya[1][1], myb[1][1], mya[1][2], myb[1][2],
			   mya[2][0], myb[2][0], mya[2][1], myb[2][1], mya[2][2], myb[2][2]);
	}

	// Output local results to compare against sequential
	for (i = 0; i < SQRP; i++)
	{
		for (j = 0; j < SQRP; j++)
		{
			printf("PAR, RANK %d: c[%d][%d] = %f\n", rank, (rank / SQRP) + i, (rank % SQRP) + j, myc[i][j]);
		}
	}

	MPI_Finalize();
}
