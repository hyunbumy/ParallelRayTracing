#pragma once

#include <vector>

#include "Resources.h"

class Output
{
public:
    static void OutputPPM(std::vector<std::vector<Vector3> > image);
    static void OutputLog(std::vector<std::vector<Vector3> > image);
};
