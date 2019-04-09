#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <stddef.h>
#include <math.h>
#include <vector>
#include <iostream>

#include "Resources.h"
#include "Scene.h"
#include "RayTracer.h"

using namespace std;

int main(int argc, char **argv) {
    struct timespec start, stop; 
    double time;

    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
  
    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        fprintf(stderr,"Requires at least two processes.\n");
        exit(-1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Scene scene("");

    if (rank == 0) {
        RayTracer::master(size, rank, scene);
        if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}   
        time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
        cout << "Execution Time: " << time << endl;
    }
    else {
        RayTracer::worker(rank, scene);
    }

    MPI_Finalize();

    return 0;
}