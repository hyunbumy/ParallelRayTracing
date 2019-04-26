#pragma once
#include <iostream>
#include <vector>
#include <iostream>
#include "Resources.h"
#include "Object.h"

template<typename T> 
class Vec3
{
public:
    Vec3() : x(0), y(0), z(0) {}
    Vec3(T xx) : x(xx), y(xx), z(xx) {}
    Vec3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}
    Vec3 operator * (const T& r) const { return Vec3(x * r, y * r, z * r); }
    Vec3 operator + (const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator - (const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    template<typename U>
    Vec3 operator / (const Vec3<U>& v) const { return Vec3(x / v.x, y / v.y, z / v.z); }
    friend Vec3 operator / (const T r, const Vec3& v)
    { return Vec3(r / v.x, r / v.y, r / v.z); }
    const T& operator [] (size_t i) const { return (&x)[i]; }
    T& operator [] (size_t i) { return (&x)[i]; }
    T length2() const{ return x * x + y * y + z * z; }
    friend Vec3 operator * (const float&r, const Vec3& v)
    { return Vec3(v.x * r, v.y * r, v.z * r); }
    friend std::ostream& operator << (std::ostream& os, const Vec3<T>& v)
    { os << v.x << " " << v.y << " " << v.z << std::endl; return os; }
	T x, y, z;
};

using Vec3f = Vec3<float>;
using Vec3b = Vec3<bool>;

const float kInfinity = std::numeric_limits<float>::max();

template<typename T = float>
class BBox
{
public:
    BBox() {}
    BBox(Vec3<T> min_, Vec3<T> max_)
    {
        bounds[0] = min_;
        bounds[1] = max_;
    } 
    //BBox& extendBy(const Vec3<T>& p)
    BBox& extendBy(T x, T y, T z)
    {
        if (x < bounds[0].x) bounds[0].x = x;
        if (y < bounds[0].y) bounds[0].y = y;
        if (z < bounds[0].z) bounds[0].z = z;
        if (x > bounds[1].x) bounds[1].x = x;
        if (y > bounds[1].y) bounds[1].y = y;
        if (z > bounds[1].z) bounds[1].z = z;
 
        return *this;
    }
    /*inline */ Vec3<T> centroid() const { return (bounds[0] + bounds[1]) * 0.5; }
    Vec3<T>& operator [] (bool i) { return bounds[i]; }
    const Vec3<T> operator [] (bool i) const { return bounds[i]; }
    bool intersect(const Vec3<T>&, const Vec3<T>&, const Vec3b&, float&) const;
    Vec3<T> bounds[2] = { kInfinity, -kInfinity };
};

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