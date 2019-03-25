#include <fstream>
#include <vector>

#include "Sphere.h"
#include "Output.h"
#include "RayTracer.h"

int main(int argc, char **argv)
{
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
    
    return 0;
}
