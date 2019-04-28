#pragma once
#include <vector>
#include "TriMesh.h"
#include <iostream>
#include <vector>
#include <memory>
#include <cstdio>
#include <fstream>
#include <cassert>
#include <functional>
#include "Resources.h"
using Mesh = TriMesh;
class AccelerationStructure 
{ 
public: 
   AccelerationStructure(std::vector<std::unique_ptr< Mesh>>& m) : meshes(std::move(m)) {} 
    virtual ~AccelerationStructure() {} 
    virtual bool intersect(const Vec3f& orig, const Vec3f& dir, const uint32_t& rayId, float& tHit) const 
    {         
        const Mesh* intersectedMesh = nullptr; 
        float t = kInfinity;
        float temp0 = kInfinity, temp1 = kInfinity;
        for (auto& mesh: meshes) {
            Vector3 vector3orig = Vector3(orig.x, orig.y, orig.z);
            Vector3 vector3dir = Vector3(dir.x, dir.y, dir.z);
            if (mesh->intersect(vector3orig, vector3dir, temp0, temp1) && t < tHit) { 
                intersectedMesh = mesh.get(); 
                tHit = t; 
            } 
        } 
 
        return (intersectedMesh != nullptr); 
    } 
protected: 
    const std::vector<std::unique_ptr< Mesh>> meshes; 
}; 