#pragma once

#include <vector>

#include "Object.h"
#include "Resources.h"
#include "Scene.h"

#define MAX_RAY_DEPTH 4

class RayTracer
{
public:
    static std::vector<std::vector<Vector3> >
        Render(const std::vector<Object*>& objects);
    static std::vector<std::vector<Vector3> >
        Render(Scene& scene);
    static void master(int &size, int &rank, Scene &scene);
    static void worker(int &rank, Scene &scene);
private:
    static Vector3 Trace(const Vector3 &rayorig, const Vector3 &raydir, 
                         const std::vector<Object*> &objects, const int &depth);
};
