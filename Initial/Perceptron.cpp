#include "header.h"

void startClassifier(ClassifierData classifier, PointData *points, int id, int numprocs, double time_start)
{
	MPI_Status status;
	double  q, t, *qArray, *weights;
	int tag = 1, solved = 0, i;
	
	weights = (double*)calloc(classifier.K + 1, sizeof(double));

	if (id == MASTER) {
		qArray = (double*)malloc(sizeof(double)* numprocs);
		checkAllocation(qArray);
	}

	// Process time offset
	t = id * classifier.dT;

	while (tag == 1 && !solved) {

		// Stop if t > tMax
		if ((t + classifier.dT) > classifier.tMax) {
			tag = 0;
			q = -1;
		} else {
			perceptron(points, classifier, weights, &solved, &q, t);
		}

		// Results from processes
		MPI_Gather(&q, 1, MPI_DOUBLE, qArray, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

		// Check q's are in QC range
		if (id == MASTER) {
			for (i = 0; i < numprocs; i++) {
				if (qArray[i] != -1) {
					t = t + i * classifier.dT;
					solved = 1;
					tag = 0;
					break;
				}
			}
		}

		// Add time to check
		if (!solved)
			t += classifier.dT * numprocs;
		MPI_Bcast(&tag, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	}

	if (solved) {
		if (id == MASTER && MASTER != i)
			MPI_Recv(weights, classifier.K + 1, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		else if (id != MASTER)
			MPI_Send(weights, classifier.K + 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
	} else if (id == MASTER) {
		t = 0;
		while ((t + classifier.dT) <= classifier.tMax)
			t += classifier.dT;

		perceptron(points, classifier, weights, &solved, &q, t);
	}

	// Write result to file
	if (id == MASTER) {

		printf("\nRuntime: %lf seconds\nt= %lf, q= %lf\nWeights: ", MPI_Wtime() - time_start, t, q);
		for (int j = 0; j <= classifier.K; j++) printf("%lf, ", weights[j]);
		printf("\n");
		fflush(stdout);

		if (!writeToFile(weights, classifier.K, t, q, solved))
			exit(1);

		free(qArray);
	}
	free(weights);
}

// Set points, classify and check queality
void perceptron(PointData *points, ClassifierData classifier, double *Weights, int *solved, double *q, double t) {
	setPoints(points, classifier.N, classifier.K, t);
	classify(classifier, points, Weights);
	*q = misses(points, classifier.N, classifier.K,  Weights) / (double)classifier.N;
	*q < classifier.qc ? *solved = 1 : *q = -1;
}

int misses(PointData *points, int numOfPoints, int  dim, double *weights) {
	int misses = 0;

#pragma omp parallel for reduction(+:count)
	for (int i = 0; i < numOfPoints; ++i)
		if (sign(points[i].currLocation, weights, dim) != points[i].sign) misses++;

	return misses;
}


int sign(double *x, double *w, int dim)
{
	double res = w[0];
	for (int i = 0; i < dim; i++)
		res += w[i + 1] * x[i];

	return (res >= 0) ? A : B;
}


void setPoints(PointData *points, int n, int dim, double t) {
#pragma omp parallel for
	for (int i = 0; i < n; i++)
		for (int j = 0; j < dim; j++)
			points[i].currLocation[j] = points[i].initLocation[j] + points[i].velocity[j] * t;
}


void classify(ClassifierData classifier, PointData* points, double *weights)
{
	int f, counter = 0;

	// Init weights
	for (int i = 0; i <= classifier.K; i++)
		weights[i] = 0;

	for (int j = 0; j < classifier.lim; j++) {

		for (int i = 0; i < classifier.N; i++, counter++) {

			f = sign(points[i].currLocation, weights, classifier.K);

			// Update weights if not classifies
			if (f != points[i].sign) {
				counter = 0;
				redefineWeights(points[i], points[i].sign, weights, classifier.alpha, classifier.K);
			}

			// All points are classified
			if (counter >= classifier.N)  return;
		}
	} 
}


// Redefines the weights according to the algorithm
void redefineWeights(PointData X, int sign, double *weights, double alpha, int dim) {
	weights[0] += alpha * sign; // bias
	for (int i = 0; i < dim; i++) {
		weights[i + 1] += alpha * sign * X.currLocation[i];
	}
}