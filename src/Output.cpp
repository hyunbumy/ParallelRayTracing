#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>

#include "Output.h"

// Output as a ppm format
void Output::OutputPPM(std::vector<std::vector<Vector3> > image)
{
    static int fileNum = 0;
    size_t height = image.size();
    if (height < 1)
        return;
    size_t width = image[0].size();

    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << fileNum;
    std::string s = ss.str();

    std::string filename = "./frames/frame" + s +  ".ppm";
    std::cout << "saving: " << filename << std::endl;

    std::ofstream ofs(filename, std::ios::out | std::ios::binary);
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
    fileNum++;
}

void Output::OutputLog(std::vector<std::vector<Vector3> > image)
{
    size_t height = image.size();
    if (height < 1)
        return;
    size_t width = image[0].size();
    std::ofstream ofs("./log", std::ios::out | std::ios::binary);
    for (unsigned i = 0; i < height; ++i)
    {
        for (unsigned j = 0; j < width; ++j)
        {
            ofs << image[i][j].x << " " << image[i][j].y << " " << image[i][j].z << std::endl;
        }
    }
    ofs.close();
}
