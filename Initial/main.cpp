#include <stdio.h>
#include "header.h"
#include "mpi.h"

int main(int argc, char* argv[])
{
	int id, numOfProcesses, bufSize;
	char *buf;
	double time_start;

	ClassifierData classifier;
	PointData *points;
	FILE *file;

	// MPI Initialization
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcesses);

	MPI_Datatype MPI_Point, MPI_Classifier;
	definePointType(&MPI_Point);
	defineClassifierType(&MPI_Classifier);

	time_start = MPI_Wtime();

	if (id == MASTER) {
		if (!readFromFile(&file, &classifier, &points)) {
			MPI_Abort(MPI_COMM_WORLD, 0);
			exit(1);
		}
	}

	bufSize = (sizeof(PointData) + (2 * sizeof(double)*classifier.K)) * classifier.N;
	buf = (char*)malloc(bufSize);

	// Broadcast classifier properties
	MPI_Bcast(&classifier, 1, MPI_Classifier, MASTER, MPI_COMM_WORLD); 

	if (id != MASTER) 
		allocatePoints(&points, classifier.K, classifier.N, 1, 0);
	else
		packPoints(points, classifier.N, buf, bufSize, classifier.K, MPI_Point);

	// Broadcast points
	MPI_Bcast(buf, bufSize, MPI_PACKED, MASTER, MPI_COMM_WORLD);

	if (id != MASTER)  unpackPoints(&points, classifier.N, buf, bufSize, classifier.K, MPI_Point);

	free(buf);
	startClassifier(classifier, points, id, numOfProcesses, time_start);

	freePoint(points, classifier.N);
	MPI_Finalize();
	return 0;
}





