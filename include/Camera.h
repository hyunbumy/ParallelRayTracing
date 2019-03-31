#pragma once

#include <vector>
#include <string>

#include "Object.h"
#include "Resources.h"

class Camera {
public:
    Vector3 position, direction;
    float fov;
    unsigned width, height;

    Camera();
    
    Camera(Vector3 pos, Vector3 dir, float f, unsigned w, unsigned h) : 
        position(pos), direction(dir), fov(f), width(w), height(h) {};
};