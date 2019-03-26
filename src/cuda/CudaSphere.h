#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

class CudaSphere
{
public:
    float3 surfaceColor, emissionColor;
    float transparency, reflection;

    float3 center;
    float radius, radius2;

    CudaSphere(const float3& c, const float& r, const float3 &sc, 
               const float &refl = 0, const float &transp = 0,
               const float3 &ec = make_float3(0, 0, 0));
    __host__ __device__
    bool Intersect(float3& rayorig, float3& raydir, float& t0, float& t1);
};
