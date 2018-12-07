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
	float a[N][N], b[N][N], c[N][N], myc[SQRP][SQRP], mya[SQRP][SQRP], myb[SQRP][SQRP], tmpdata;
	MPI_Request sendreq, rcvreq;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// sequential MM
	if (rank == 0)
	{

		// read in matrix
		f = fopen("matrixA.dat", "r");
		for (i = 0; i < N; i++)
		{
			for (j = 0; j < N; j++)
			{
				error = fscanf(f, "%f", &tmpdata);
				a[i][j] = tmpdata;
				b[i][j] = tmpdata + 1.;
			}
		}
		fclose(f);

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
				printf("SEQ: c[%d][%d] = %f\n", i, j, c[i][j]);
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
	int y = coords[0];

	// Scatter a and b
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

	for (i = 0; i < SQRP; i++)
	{
		for (j = 0; j < SQRP; j++)
		{
			myc[i][j] 0.0;
		}
	}

	MPI_Isend(mya, N * N / P, MPI_FLOAT, (rank + 1) % P, 0, MPI_COMM_WORLD, &sendreq);
	int src = (rank == 0) ? P - 1 : rank - 1;
	MPI_Irecv(tmpa, N * N / P, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &rcvreq);
	… MPI_Wait(&rcvreq, &status);

	// TODO: Move a and b data within Cartesian Grid using initial skew
	// operations (see p. 10 of Lecture 20.)

	// TODO: Add following loop:
	// for (k=0; k<=SQRP-1; k++} {
	//    CALC: Should be like sequential code, but use
	//          myc, mya, and myb.  Adjust bounds for all loops to SQRP.
	//          (More generally, (N/P/SQRP)).
	//    SHIFT: Shift A leftward and B upward by 1 in Cartesian grid.
	// }

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
