#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <cmath>

#define MASTER 0 // Master rank

enum Classes { A = 1, B = -1 };

typedef struct {
	int N;			// Num of points
	int K;			// Num of coordinates
	double dT;		// Increment value of t
	double tMax;	// Max t
	double alpha;	// Conversion ratio
	int lim;		// Max num of iterations
	double qc;		// Quality of Classifier 
}ClassifierData;

typedef struct {
	double *initLocation;	// Initial lcation vector
	double *velocity;		// Velocities vector
	double *currLocation;	// Current locations vector
	int sign;
}PointData;

void definePointType(MPI_Datatype* MPI_Point);
void defineClassifierType(MPI_Datatype* MPI_Classifier);

int readFromFile(FILE **F, ClassifierData * C, PointData ** P);
int writeToFile(double *weights, int dim, double time, double q, int solved);
void readPoints(FILE *F, PointData **P, int dim, int n);
void allocatePoints(PointData **p, int dim, int n, int includeBase, int includeVectors);
void packPoints(PointData *points, int n, char* buffer, int bufferSize, int dim, MPI_Datatype MPI_Point);
void unpackPoints(PointData **p, int n, char* buffer, int bufferSize, int dim, MPI_Datatype MPI_Point);
void freePoint(PointData *p, int n);
void checkAllocation(void *p);

void classify(ClassifierData C, PointData* points, double *weights);
void setPoints(PointData *points, int n, int dim, double t);
int sign(double *x, double *w, int dim);
int misses(PointData *points, int numOfPoints, int  dim, double *weights);
void startClassifier(ClassifierData classifier, PointData *points, int id, int numprocs, double time0);
void perceptron(PointData *points, ClassifierData classifier, double *Weights, int *solved, double *q, double t);
void redefineWeights(PointData X, int sign, double *weights, double alpha, int dim);