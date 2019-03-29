#include <fstream>
#include <iostream>

#include "CudaRT.h"
#include "CudaRTIter.h"

// Output as a ppm format
void OutputPPM(float3* image, unsigned width, unsigned height)
{
    std::ofstream ofs("./build/untitled.ppm", std::ios::out | std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (unsigned i = 0; i < height*width; ++i)
    {
        ofs << (unsigned char)(std::min(float(1), image[i].x) * 255) <<
               (unsigned char)(std::min(float(1), image[i].y) * 255) <<
               (unsigned char)(std::min(float(1), image[i].z) * 255);
    }
    ofs.close();
}

void OutputLog(float3* image, unsigned width, unsigned height)
{
    std::ofstream ofs("./build/log", std::ios::out | std::ios::binary);
    for (unsigned i = 0; i < height*width; ++i)
    {
        ofs << image[i].x << " " << image[i].y << " " << image[i].z << std::endl;
    }
    ofs.close();
}

int main()
{
    struct timespec start, stop; 
    double time;

    unsigned width = 3280, height = 2160;

    // Allocate image buffer
    float3* image = new float3[width*height];

    // Run GPU Ray Tracer
    // std::cout << "Start GPU" << std::endl;
    // CudaRT::RenderWrapper(image, width, height);
    // std::cout << "Finish GPU" << std::endl;

    // OutputPPM(image, width, height);
    // OutputLog(image, width, height);
    // std::cout << "Outputed" << std::endl;

    // Run GPU Iterative RT
    std::cout << "Start Iterative RT" << std::endl;
    if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
    
    // Iterative CUDA kernel
    CudaRTIter rt(width, height);
    rt.RenderWrapper(image);
    std::cout << "Finish Iterative RT" << std::endl;

    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}   
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    std::cout << "\nExecution Time: " << time << std::endl;
    
    OutputPPM(image, width, height);
    // OutputLog(image, width, height);

    delete image;

    return 0;
}
