

#include <stdio.h>

#include <omp.h>

static long num_steps = 0x1<<16;

static int num_threads = 64;

double  parallel() {
	int i;
	double x, pi, sum = 0.0;
	double step = 1.0/ (double)num_steps;

	double t1 = omp_get_wtime();

	omp_set_num_threads(num_threads);
	#pragma omp parallel for reduction(+:sum) private(x)
	for (i=0; i<num_steps; i++) {
		x = (i+.5)*step;
		sum += 4.0 / (1.0+x*x);
	}
	pi = step * sum;

	double t2 = omp_get_wtime();

	printf("\n*** Parallel ***\n");
	printf("Pi = %f\n", pi);
	printf("t  = %f\n", t2-t1);
	
	return t2-t1;
}


double sequential() {
        int i;
        double x, pi, sum = 0.0;
        double step = 1.0/ (double)num_steps;

        double t1 = omp_get_wtime();

        for (i=0; i<num_steps; i++) {
                x = (i+.5)*step;
                sum += 4.0 / (1.0+x*x);
        }
        pi = step * sum;

        double t2 = omp_get_wtime();

        printf("\n*** Sequential ***\n");
        printf("Pi = %f\n", pi);
        printf("t  = %f\n", t2-t1);

	return t2-t1;
}


int main() {
        double tseq = sequential();
        double tpar = parallel();

	double speedUp = tseq / tpar;
	double eff = speedUp / (double) num_threads;

	printf("\n### Metrics ###\n");
	printf("speedUp = %f\n", speedUp);
	printf("threads = %i\n", num_threads);
	printf("Efficiency = %f\n\n", eff); 
}
