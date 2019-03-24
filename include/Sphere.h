#pragma once

#include "Math.h"

class Sphere
{
public:
    Vector3 center;                           /// position of the sphere
    float radius, radius2;                  /// sphere radius and radius^2
    Vector3 surfaceColor, emissionColor;      /// surface color and emission (light)
    float transparency, reflection;         /// surface transparency and reflectivity
    Sphere(
        const Vector3 &c,
        const float &r,
        const Vector3 &sc,
        const float &refl = 0,
        const float &transp = 0,
        const Vector3 &ec = Vector3::Zero) :
        center(c), radius(r), radius2(r * r), surfaceColor(sc), emissionColor(ec),
        transparency(transp), reflection(refl)
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
};