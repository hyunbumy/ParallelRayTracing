#pragma once

#include <vector>
#include <string>

#include "Object.h"
#include "Resources.h"

class Camera {
public:
    Vector3 position, direction, movement;
    float iterations, fov;
    unsigned width, height;

    Camera();
    
    Camera(Vector3 pos, Vector3 dir, Vector3 movement, float iterations, float f, unsigned w, unsigned h) : 
        position(pos), direction(dir), movement(movement), iterations(iterations), fov(f), width(w), height(h) {};
};