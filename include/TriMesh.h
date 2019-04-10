#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include "Resources.h"
#include "Object.h"
#include "Triangle.h"

class TriMesh : public Object
{
public:
    std::vector<Triangle> triangles;
    int intersectIndex = 0;

    TriMesh(
        const char *file,
        const Vector3 &sc,
        const float &refl = 0,
        const float &transp = 0,
        const Vector3 &ec = Vector3::Zero) : Object(sc, refl, transp, ec)
    {
        std::ifstream ifs;
        try {

            std::string test(file);
            ifs.open(file);
            if (ifs.fail()) throw;
            std::stringstream ss;
            ss << ifs.rdbuf();

            if(test.find(".geo") != std::string::npos) {
                uint32_t numVerticies;
                ss >> numVerticies;

                //std::cout << "numVerticies: " << numVerticies << std::endl;

                std::vector<Vector3> verticies;
                for(uint32_t i = 0; i < numVerticies; i++) {
                    Vector3 temp;
                    ss >> temp.x;
                    ss >> temp.y;
                    ss >> temp.z;
                    //std::cout << temp.x << " " << temp.y << " " << temp.z << std::endl;
                    verticies.push_back(temp);
                }

                uint32_t numTriangles;
                ss >> numTriangles;

                //std::cout << "numTriangles: " << numTriangles << std::endl;

                for(uint32_t i = 0; i  < numTriangles; i++) {
                    int v0, v1, v2;
                    ss >> v0;
                    ss >> v1;
                    ss >> v2;
                    //std::cout << v0 << " " << v1 << " " << v2 << std::endl;
                    triangles.push_back(Triangle(verticies[v0], verticies[v1], verticies[v2], sc, refl, transp));
                }
            } else if(test.find(".dae") != std::string::npos) {
                uint32_t numVerticies;
                ss >> numVerticies;

                //std::cout << "numVerticies: " << numVerticies << std::endl;

                std::vector<Vector3> verticies;
                for(uint32_t i = 0; i < numVerticies; i++) {
                    Vector3 temp;
                    ss >> temp.x;
                    ss >> temp.y;
                    ss >> temp.z;
                    //std::cout << temp.x << " " << temp.y << " " << temp.z << std::endl;
                    verticies.push_back(temp);
                }

                                uint32_t numTriangles;
                ss >> numTriangles;

                //std::cout << "numTriangles: " << numTriangles << std::endl;

                for(uint32_t i = 0; i  < numTriangles; i++) {
                    int v0, v1, v2, garbage;
                    ss >> v0;
                    ss >> garbage;
                    ss >> garbage;
                    ss >> v1;
                    ss >> garbage;
                    ss >> garbage;
                    ss >> v2;
                    ss >> garbage;
                    ss >> garbage;
                    //std::cout << v0 << " " << v1 << " " << v2 << std::endl;
                    triangles.push_back(Triangle(verticies[v0], verticies[v1], verticies[v2], sc, refl, transp));
                }
                std::cout << "finished parsing dae" << std::endl;

            }
        } catch (...) {
            std::cout << "error getting file" << std::endl;
            ifs.close();
        }
        ifs.close();
    }

    bool intersect(const Vector3 &rayorig, const Vector3 &raydir, float &t0, float &t1)
    {
        bool returnBool = false;

        for(size_t i = 0; i < triangles.size(); i++) {
            float temp0 = INFINITY, temp1 = INFINITY;
            if(triangles[i].intersect(rayorig, raydir, temp0, temp1) && temp0 < t0) {
                returnBool = true;
                t0 = temp0;
                intersectIndex = i;
            }
        }
        //std::cout << "intersect with trimesh? " << returnBool << std::endl;

        return returnBool;
    }

    Vector3 calculateHit(const Vector3 &rayorig) const {
        return triangles[intersectIndex].calculateHit(rayorig);
    }
};