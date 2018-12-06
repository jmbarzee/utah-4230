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

void
main(int argc, char *argv[]) {
  FILE *f;
  int i, j, k, error, rank, size;
  float a[N][N], b[N][N], c[N][N], myc[SQRP][SQRP], mya[SQRP][SQRP], myb[SQRP][SQRP], tmpdata;
  MPI_Request sendreq, rcvreq;
  MPI_Status status;


   MPI_Init(&argc, &argv);
   MPI_Comm_rank( MPI_COMM_WORLD, &rank );
   MPI_Comm_size( MPI_COMM_WORLD, &size );


   // sequential MM 
   if (rank == 0) {

     // read in matrix
     f = fopen("matrixA.dat","r");
     for (i = 0; i<N; i++) {   
      for (j = 0; j<N; j++) {   
        error = fscanf(f,"%f",&tmpdata);
        a[i][j] = tmpdata;
        b[i][j] = tmpdata+1.;
      }
     }
     fclose(f);  

     // After computing each point, output sequential results.
     for (i = 0; i< N; i++) {
       for (j = 0; j<N; j++) {
         c[i][j] = 0.;
         for (k=0; k<N; k++) {
           c[i][j] += a[i][k] * b[k][j];
         }
         printf("SEQ: c[%d][%d] = %f\n", i,j,c[i][j]);
        }
     }
   }

   // TODO: Parallel Portion.  Distribute a and b into local copies 
   // mya and myb using Scatterv as in website pointed to by Lecture 21.  
   // Initialize myc to 0.  

   // TODO: Now create Cartesian grid communicator (see website pointed to
   // by Lecture 21 and Sundar-communicators.pdf on Canvas)

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
   for (i = 0; i<SQRP; i++) {   
      for (j = 0; j<SQRP; j++) {   
	printf("PAR, RANK %d: c[%d][%d] = %f\n", rank, (rank/SQRP)+i,(rank % SQRP)+j,myc[i][j]);
      }
   }

   MPI_Finalize();
}

