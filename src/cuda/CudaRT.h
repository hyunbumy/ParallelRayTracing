#pragma once

#include "CudaCommon.h"

class CudaRT
{
public:
    static void RenderWrapper(float3* image, unsigned width, unsigned height);
};
