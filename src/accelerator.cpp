#include "Resources.h"
#include "Object.h"
#include <vector>
#include <iostream>
#include "accelerator.h"
#ifndef M_PI 
#define M_PI (3.14159265358979323846) 
#endif 

template<typename T>
Vec3<T> cross(const Vec3<T>& a, const Vec3<T>& b) 
{
    return Vec3<T>(a.y * b.z - a.z * b.y, 
                   a.z * b.x - a.x * b.z, 
                   a.x * b.y - a.y * b.x); 
}

template<typename T> 
T dot(const Vec3<T>& va, const Vec3<T>& vb) 
{ return va.x * vb.x + va.y * vb.y + va.z * vb.z; } 

template<typename T> 
void normalize(Vec3<T>& vec)
{
    T len2 = vec.length2(); 
    if (len2 > 0) {
        T invLen = 1 / sqrt(len2);
        vec.x *= invLen, vec.y *= invLen, vec.z *= invLen;
    }
}

template<typename T> 
bool BBox<T>::intersect(const Vec3<T>& orig, const Vec3<T>& invDir, const Vec3b& sign, float& tHit) const 
{
    numRayBBoxTests++;
    float tmin, tmax, tymin, tymax, tzmin, tzmax;

    tmin  = (bounds[sign[0]    ].x - orig.x) * invDir.x;
    tmax  = (bounds[1 - sign[0]].x - orig.x) * invDir.x;
    tymin = (bounds[sign[1]    ].y - orig.y) * invDir.y;
    tymax = (bounds[1 - sign[1]].y - orig.y) * invDir.y;

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    tzmin = (bounds[sign[2]    ].z - orig.z) * invDir.z;
    tzmax = (bounds[1 - sign[2]].z - orig.z) * invDir.z;

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;

    tHit = tmin;

    return true;
} 