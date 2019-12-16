#include "header.h"

void definePointType(MPI_Datatype* MPI_Point) {
	int blockLengths[4] = { 1, 1, 1, 1 };
	MPI_Aint disp[4];
	MPI_Datatype types[4] = { MPI_DOUBLE,MPI_DOUBLE, MPI_DOUBLE, MPI_INT };

	disp[0] = offsetof(PointData, initLocation);
	disp[1] = offsetof(PointData, velocity);
	disp[2] = offsetof(PointData, currLocation);
	disp[3] = offsetof(PointData, sign);

	MPI_Type_create_struct(4, blockLengths, disp, types, MPI_Point);
	MPI_Type_commit(MPI_Point);
}

void defineClassifierType(MPI_Datatype* MPI_Classifier) {
	int blockLengths[7] = { 1, 1, 1, 1, 1, 1, 1 };
	MPI_Aint disp[7];
	MPI_Datatype types[74] = { MPI_INT, MPI_INT, MPI_DOUBLE,MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_DOUBLE };

	disp[0] = offsetof(ClassifierData, N);
	disp[1] = offsetof(ClassifierData, K);
	disp[2] = offsetof(ClassifierData, dT);
	disp[3] = offsetof(ClassifierData, tMax);
	disp[4] = offsetof(ClassifierData, alpha);
	disp[5] = offsetof(ClassifierData, lim);
	disp[6] = offsetof(ClassifierData, qc);

	MPI_Type_create_struct(7, blockLengths, disp, types, MPI_Classifier);
	MPI_Type_commit(MPI_Classifier);
}

int readFromFile(FILE **file, ClassifierData * classifier, PointData ** P) {

	fflush(stdout);
	fopen_s(file, "E:/mpi/Input.txt", "r+");

	if (*file == NULL) {
		perror("E:/mpi/Input.txt");
		fflush(stdout);
		return 0;
	}
	else fflush(stdout);

	fscanf_s(*file, "%d %d %lf %lf %lf %d %lf", &classifier->N, &classifier->K, &classifier->dT, &classifier->tMax, &classifier->alpha, &classifier->lim, &classifier->qc);
	readPoints(*file, P, classifier->K, classifier->N);

	fclose(*file);
	fflush(stdout);
	return 1;
}

int writeToFile(double *weights, int dim, double time, double q, int solved) {
	FILE *f;
	fopen_s(&f, "E:/mpi/Output.txt", "w");

	if (f == NULL) {
		printf("Error writing to file!\n");
		fflush(stdout);
		return 0;
	}

	solved ? fprintf(f, "time minimum = %lf	q= %lf\n\n", time, q) : fprintf(f, "time was not found\n\n");

	for (int i = 0; i <= dim; i++) fprintf(f, "%lf\n", weights[i]);

	fclose(f);
	fflush(stdout);
	return 1;
}

void readPoints(FILE *file, PointData **P, int dim, int numOfPoints) {

	allocatePoints(P, dim, numOfPoints, 1, 1);
	PointData *points = (*P);

	for (int i = 0; i < numOfPoints; i++) {
		for (int j = 0; j < dim; j++) fscanf_s(file, "%lf", &points[i].initLocation[j]);
		for (int j = 0; j < dim; j++) fscanf_s(file, "%lf", &points[i].velocity[j]);
		fscanf_s(file, "%d", &points[i].sign);
	}
	fflush(stdout);
}

void allocatePoints(PointData **p, int dim, int numOfPoints, int allocPoints, int allocVectors) {

	PointData *points;
	double *vec_a, *vec_b, *vec_c;

	// Allocate Points
	if (allocPoints) { 
		*p = (PointData*)malloc(sizeof(PointData)*numOfPoints);
		checkAllocation(*p);
	}

	points = *p;

	// Allocate vectors
	if (allocVectors) { 

		vec_a = (double*)malloc(sizeof(double) * numOfPoints * dim);
		vec_b = (double*)malloc(sizeof(double) * numOfPoints * dim);
		vec_c = (double*)malloc(sizeof(double) * numOfPoints * dim);
		
		// Set vectors
		for (int i = 0; i < numOfPoints; i++) {
			points[i].initLocation = vec_a + i * dim;
			points[i].velocity = vec_b + i * dim;
			points[i].currLocation = vec_c + i * dim;

			checkAllocation(points[i].initLocation);
			checkAllocation(points[i].velocity);
			checkAllocation(points[i].currLocation);
		}
	}
}


void packPoints(PointData *points, int n, char* buffer, int bufferSize, int dim, MPI_Datatype MPI_Point) {
	int position = 0;
	MPI_Pack(points, n, MPI_Point, buffer, bufferSize, &position, MPI_COMM_WORLD);
	MPI_Pack(points->initLocation, dim * n, MPI_DOUBLE, buffer, bufferSize, &position, MPI_COMM_WORLD);
	MPI_Pack(points->velocity, dim * n, MPI_DOUBLE, buffer, bufferSize, &position, MPI_COMM_WORLD);
}

void unpackPoints(PointData **p, int numOfPoints, char* buffer, int bufferSize, int dim, MPI_Datatype MPI_Point) {
	int position = 0;
	PointData *points = *p;

	allocatePoints(p, dim, numOfPoints, 0, 1);
	MPI_Unpack(buffer, bufferSize, &position, points, numOfPoints, MPI_Point, MPI_COMM_WORLD);
	MPI_Unpack(buffer, bufferSize, &position, points->initLocation, dim * numOfPoints, MPI_DOUBLE, MPI_COMM_WORLD);
	MPI_Unpack(buffer, bufferSize, &position, points->velocity, dim * numOfPoints, MPI_DOUBLE, MPI_COMM_WORLD);
}


void freePoint(PointData *p, int n) {
	free(p->initLocation);
	free(p->velocity);
	free(p->currLocation);
	free(p);
}

void checkAllocation(void *p) {
	if (!p) {
		printf("MEMORY ALLOCATION ERROR!\n");
		fflush(stdout);
		exit(1);
	}
}