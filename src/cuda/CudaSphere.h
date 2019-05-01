#pragma once

#include "CudaObject.h"

class CudaSphere : public CudaObject
{
public:
    float3 surfaceColor, emissionColor;
    float transparency, reflection;

    float3 center;
    float radius, radius2;

    __device__
    CudaSphere(const float3& c, const float& r, const float3 &sc, 
               const float &refl = 0, const float &transp = 0,
               const float3 &ec = make_float3(0, 0, 0));
    __device__
    bool Intersect(const float3& rayorig, const float3& raydir, float& t0, float& t1) override;
    __device__
    float3 CalculateHit(const float3 &rayorig) const override;
};
