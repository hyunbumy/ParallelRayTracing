#pragma once

#include "Math.h"
#include "Object.h"

class Sphere : public Object
{
public:
    Vector3 center;                           /// position of the sphere
    float radius, radius2;                  /// sphere radius and radius^2
    Sphere(
        const Vector3 &c,
        const float &r,
        const Vector3 &sc,
        const float &refl = 0,
        const float &transp = 0,
        const Vector3 &ec = Vector3::Zero) : Object(sc, refl, transp, ec), 
        center(c), radius(r), radius2(r * r)
    { /* empty */ }

    ~Sphere()
    { /* empty */ }
    
    //[comment]
    // Compute a ray-sphere intersection using the geometric solution
    //[/comment]
    bool intersect(const Vector3 &rayorig, const Vector3 &raydir, float &t0, float &t1) const
    {
        Vector3 l = center - rayorig;
        float tca = Vector3::Dot(l, raydir);
        if (tca < 0) return false;
        float d2 = Vector3::Dot(l, l) - tca * tca;
        if (d2 > radius2) return false;
        float thc = sqrt(radius2 - d2);
        t0 = tca - thc;
        t1 = tca + thc;
        
        return true;
    }

    Vector3 calculateHit(const Vector3 &rayorig) const {
        return rayorig - this->center;
    }
};