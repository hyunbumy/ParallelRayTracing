#include <fstream>
#include <vector>

#include "Sphere.h"
#include "RayTracer.h"

// Output as a ppm format
void output(std::vector<std::vector<Vector3>> image)
{
    auto height = image.size();
    if (height < 1)
        return;
    auto width = image[0].size();
    std::ofstream ofs("./untitled.ppm", std::ios::out | std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (unsigned i = 0; i < height; ++i)
    {
        for (unsigned j = 0; j < width; ++j)
        {
            ofs << (unsigned char)(std::min(float(1), image[i][j].x) * 255) <<
                   (unsigned char)(std::min(float(1), image[i][j].y) * 255) <<
                   (unsigned char)(std::min(float(1), image[i][j].z) * 255);
        }
    }
    ofs.close();
}

int main(int argc, char **argv)
{
    srand48(13);
    std::vector<Sphere> spheres;
    // position, radius, surface color, reflectivity, transparency, emission color
    for (int i = 0; i < 1; ++i)
    {
    spheres.push_back(Sphere(Vector3( 0.0, -10004, -20), 10000, Vector3(0.20, 0.20, 0.20), 0, 0.0));
    spheres.push_back(Sphere(Vector3( 0.0,      0, -20),     4, Vector3(1.00, 0.32, 0.36), 1, 0.5));
    spheres.push_back(Sphere(Vector3( 5.0,     -1, -15),     2, Vector3(0.90, 0.76, 0.46), 1, 0.0));
    spheres.push_back(Sphere(Vector3( 5.0,      0, -25),     3, Vector3(0.65, 0.77, 0.97), 1, 0.0));
    spheres.push_back(Sphere(Vector3(-5.5,      0, -15),     3, Vector3(0.90, 0.90, 0.90), 1, 0.0));
    }
    // light
    spheres.push_back(Sphere(Vector3( 0.0,     20, -30),     3, Vector3(0.00, 0.00, 0.00), 0, 0.0, Vector3(3,3,3)));

    output(RayTracer::Render(spheres));
    
    return 0;
}
