#pragma once

#include "Resources.h"
#include "Object.h"
#include <iostream>

class Triangle : public Object
{
public:
    Vector3 v0, v1, v2;
    Triangle(
        const Vector3 &v0,
        const Vector3 &v1,
        const Vector3 &v2,
        const Vector3 &sc,
        const float &refl = 0,
        const float &transp = 0,
        const Vector3 &ec = Vector3::Zero) : Object(sc, refl, transp, ec), 
        v0(v0), v1(v1), v2(v2)
    { /* empty */ }

    ~Triangle()
    { /* empty */ }
    
    //[comment]
    // Compute a ray-sphere intersection using the geometric solution
    //[/comment]
    bool intersect(const Vector3 &rayorig, const Vector3 &raydir, float &t0, float &t1)
    {
        float u = INFINITY, v = INFINITY;

        Vector3 v0v1 = v1 - v0;
        Vector3 v0v2 = v2 - v0;
        Vector3 pvec = Vector3::Cross(raydir, v0v2);
        float det = Vector3::Dot(v0v1, pvec);

        // ray and triangle are parallel if det is close to 0
        if (fabs(det) < 1e-8) return false;

        float invDet = 1 / det;

        Vector3 tvec = rayorig - v0;
        u = Vector3::Dot(tvec, pvec) * invDet;
        if (u < 0 || u > 1) return false;

        Vector3 qvec = Vector3::Cross(tvec, v0v1);
        v = Vector3::Dot(raydir, qvec) * invDet;
        if (v < 0 || u + v > 1) return false;
        
        t0 = Vector3::Dot(v0v2, qvec) * invDet;
        
        return true;
    }

    Vector3 calculateHit(const Vector3 &rayorig) const {

        return Vector3::Cross(v1 - v0, v2 - v0);
    }
};