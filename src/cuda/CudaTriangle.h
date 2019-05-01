#pragma once

#include "CudaObject.h"

class CudaTriangle: public CudaObject
{
public:
    float3 v0, v1, v2;

    __device__
    CudaTriangle(const float3 &v0, const float3 &v1, const float3& v2,
                 const float3& sc, const float& refl=0, const float& transp=0,
                 const float3& ec = make_float3(0,0,0));

    __device__
    bool Intersect(const float3& rayorig, const float3& raydir,
                   float& t0, float& t1) override;

    __device__
    float3 CalculateHit(const float3 &rayorig) const override;
};
