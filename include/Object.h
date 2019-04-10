#pragma once

#include "Resources.h"

class Object
{
public:

    Vector3 surfaceColor, emissionColor;
    float transparency, reflection;

    Object(
        const Vector3 &sc,
        const float &refl = 0,
        const float &transp = 0,
        const Vector3 &ec = Vector3::Zero) :
        surfaceColor(sc), emissionColor(ec), transparency(transp), reflection(refl)
    { /* empty */ }

    virtual ~Object()
    { /* empty */ }
    
    virtual bool intersect(const Vector3 &rayorig, const Vector3 &raydir, float &t0, float &t1) = 0;
    virtual Vector3 calculateHit(const Vector3 &rayorig) const = 0;
};