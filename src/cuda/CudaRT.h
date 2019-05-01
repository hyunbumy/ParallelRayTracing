#pragma once

#include <vector>
#include <string>
#include "CudaCommon.h"

class CudaRT
{
public:
    static void RenderWrapper(float3* image, unsigned width, unsigned height);
    static int ParseTriMesh(std::vector<std::string> files, float3*& triangles);
};
