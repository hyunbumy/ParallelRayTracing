#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

class CudaObject
{
public:

    float3 surfaceColor, emissionColor;
    float transparency, reflection;

    __device__
    CudaObject(
        const float3 &sc,
        const float &refl = 0,
        const float &transp = 0,
        const float3 &ec = make_float3(0,0,0)) :
        surfaceColor(sc), emissionColor(ec), transparency(transp), reflection(refl)
    { /* empty */ }

    __device__
    virtual ~CudaObject()
    { /* empty */ }
    
    __device__
    virtual bool Intersect(const float3 &rayorig, const float3 &raydir, float &t0, float &t1) = 0;
    __device__
    virtual float3 CalculateHit(const float3 &rayorig) const = 0;
};