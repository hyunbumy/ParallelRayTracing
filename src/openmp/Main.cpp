#include <fstream>
#include <vector>
#include <iostream>

#include "Sphere.h"
#include "Output.h"
#include "RayTracer.h"

using namespace std;

int main(int argc, char **argv)
{
    struct timespec start, stop; 
    double time;

    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
    //srand48(13);
    std::vector<Object*> objects;
    // position, radius, surface color, reflectivity, transparency, emission color
    objects.push_back(new Sphere(Vector3( 0.0, -10004, -20), 10000, Vector3(0.20, 0.20, 0.20), 0, 0.0));
    objects.push_back(new Sphere(Vector3( 0.0,      0, -20),     4, Vector3(1.00, 0.32, 0.36), 1, 0.5));
    objects.push_back(new Sphere(Vector3( 5.0,     -1, -15),     2, Vector3(0.90, 0.76, 0.46), 1, 0.0));
    objects.push_back(new Sphere(Vector3( 5.0,      0, -25),     3, Vector3(0.65, 0.77, 0.97), 1, 0.0));
    objects.push_back(new Sphere(Vector3(-5.5,      0, -15),     3, Vector3(0.90, 0.90, 0.90), 1, 0.0));
    // light
    objects.push_back(new Sphere(Vector3( 0.0,     20, -30),     3, Vector3(0.00, 0.00, 0.00), 0, 0.0, Vector3(3,3,3)));

    Output::OutputPPM(RayTracer::Render(objects));

    for(auto object : objects) delete object;

    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}   
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    cout << "Execution Time: " << time << endl;
    
    return 0;
}
