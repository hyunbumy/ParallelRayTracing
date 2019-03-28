#pragma once

#include <vector>
#include <string>

#include "Object.h"
#include "Math.h"
#include "Camera.h"


class Scene{
public:
    Camera cam;
    std::vector<Object*> Objects;
    Scene(std::string filename);
    ~Scene();
};
