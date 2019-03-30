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
    void Allocate(unsigned width, unsigned height);
    void Deallocate(unsigned width, unsigned height);

    float mWidth;
    float mHeight;
};