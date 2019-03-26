#include <fstream>
#include <iostream>

#include "CudaRT.h"

// Output as a ppm format
void OutputPPM(float* image, unsigned width, unsigned height)
{
    std::ofstream ofs("./untitled.ppm", std::ios::out | std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (unsigned i = 0; i < height*width*3; i+=3)
    {
        ofs << (unsigned char)(std::min(float(1), image[i]) * 255) <<
               (unsigned char)(std::min(float(1), image[i+1]) * 255) <<
               (unsigned char)(std::min(float(1), image[i+2]) * 255);
    }
    ofs.close();
}

int main()
{
    unsigned width = 640, height = 480;

    // Allocate image buffer
    float* image = new float[width*height*3];

    // Run GPU Ray Tracer
    std::cout << "Start GPU" << std::endl;
    CudaRT::RenderWrapper(image, width, height);
    std::cout << "Finish GPU" << std::endl;

    OutputPPM(image, width, height);
    std::cout << "Outputed" << std::endl;

    delete image;

    return 0;
}
