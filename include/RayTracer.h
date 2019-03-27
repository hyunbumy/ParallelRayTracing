#pragma once

#include <vector>

#include "Object.h"
#include "Math.h"

class RayTracer
{
public:
    static std::vector<std::vector<Vector3> >
        Render(const std::vector<Object*>& objects);
private:
    static Vector3 Trace(const Vector3 &rayorig, const Vector3 &raydir, 
                         const std::vector<Object*> &objects, const int &depth);
};
