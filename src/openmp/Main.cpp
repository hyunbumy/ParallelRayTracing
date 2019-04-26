#include <fstream>
#include <vector>
#include <iostream>

#include "Sphere.h"
#include "Output.h"
#include "RayTracer.h"
#include "Scene.h"

using namespace std;

int main(int argc, char **argv)
{
    struct timespec start, stop; 
    double time;

    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
    //srand48(13);

    int sceneNumber = 1;

    if(argc != 1) {
        sceneNumber = std::stoi(argv[1]);
    }

    Scene scene("", sceneNumber);

    for(int i = 0; i < scene.cam.iterations; i++) {
        auto output = RayTracer::Render(scene);
        Output::OutputPPM(output);
        scene.cam.position += scene.cam.movement;
    }
    

    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}   
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    cout << "Execution Time: " << time << endl;

    //Output::OutputLog(output);
    
    
    return 0;
}
