#include <fstream>

#include "Output.h"

// Output as a ppm format
void Output::OutputPPM(std::vector<std::vector<Vector3>> image)
{
    auto height = image.size();
    if (height < 1)
        return;
    auto width = image[0].size();
    std::ofstream ofs("./build/untitled.ppm", std::ios::out | std::ios::binary);
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
