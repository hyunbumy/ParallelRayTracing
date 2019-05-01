#include "CudaTriangle.h"
#include "helper_math.h"

__device__
CudaTriangle::CudaTriangle(const float3 &v0, const float3 &v1, const float3& v2,
                           const float3& sc, const float& refl, 
                           const float& transp, const float3& ec)
    :CudaObject(sc, refl, transp, ec)
    ,v0(v0)
    ,v1(v1)
    ,v2(v2)
{   }

__device__
bool CudaTriangle::Intersect(const float3& rayorig, const float3& raydir,
                             float& t0, float& t1)
{
    float u = INFINITY, v = INFINITY;

    float3 v0v1 = v1 - v0;
    float3 v0v2 = v2 - v0;
    float3 pvec = cross(raydir, v0v2);
    float det = dot(v0v1, pvec);

    // // ray and triangle are parallel if det is close to 0
    if (fabsf(det) < 1e-8) return false;

    float invDet = 1 / det;

    float3 tvec = rayorig - v0;
    u = dot(tvec, pvec) * invDet;
    if (u < 0 || u > 1) return false;

    float3 qvec = cross(tvec, v0v1);
    v = dot(raydir, qvec) * invDet;
    if (v < 0 || u + v > 1) return false;
    
    t0 = dot(v0v2, qvec) * invDet;
    
    return true;
}

__device__
float3 CudaTriangle::CalculateHit(const float3& rayorig) const
{
    return cross(v1 - v0, v2 - v0);
}
