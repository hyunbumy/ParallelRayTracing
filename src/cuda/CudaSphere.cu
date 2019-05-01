#include "CudaSphere.h"
#include "helper_math.h"

__device__
CudaSphere::CudaSphere(const float3& c, const float& r, const float3& sc,
                       const float& refl, const float& transp, const float3& ec)
    :CudaObject(sc, refl, transp, ec)
    ,center(c)
    ,radius(r)
    ,radius2(r*r)
{   }

__device__
bool CudaSphere::Intersect(const float3& rayorig, const float3& raydir,
                           float& t0, float& t1)
{
    float3 l = center - rayorig;
    float tca = dot(l, raydir);
    if (tca < 0) return false;
    float d2 = dot(l, l) - tca * tca;
    if (d2 > radius2) return false;
    float thc = sqrtf(radius2 - d2);
    t0 = tca - thc;
    t1 = tca + thc;
    
    return true;
}

__device__
float3 CudaSphere::CalculateHit(const float3 &rayorig) const
{
    return rayorig - this->center;
}
