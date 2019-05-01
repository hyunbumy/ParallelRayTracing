#pragma once

#include "CudaCommon.h"

class CudaRTIter
{
public:
    float3* mLayers;
    float3* mDirections;
    float3* mBases;
    float3* mCoefficients;
    CudaRTIter(unsigned width, unsigned height);
    ~CudaRTIter();
    void RenderWrapper(float3* image);
private:
    float mWidth;
    float mHeight;
};