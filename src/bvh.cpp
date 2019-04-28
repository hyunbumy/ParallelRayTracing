#include "bvh.h"
#include "accelerator.h"
#include <functional>
#include <queue>

bool BVH::intersect(const Vec3f& orig, const Vec3f& dir, const uint32_t& rayId, float& tHit) const 
{ 
    tHit = kInfinity; 
    const Mesh* intersectedMesh = nullptr; 
    float precomputedNumerator[BVH::kNumPlaneSetNormals]; 
    float precomputedDenominator[BVH::kNumPlaneSetNormals]; 
    for (uint8_t i = 0; i < kNumPlaneSetNormals; ++i) { 
        precomputedNumerator[i] = dot(planeSetNormals[i], orig); 
        precomputedDenominator[i] = dot(planeSetNormals[i], dir); 
    } 
 
    /* 
    tNear = kInfinity; // set 
    for (uint32_t i = 0; i < meshes.size(); ++i) { 
        numRayVolumeTests++; 
        float tn = -kInfinity, tf = kInfinity; 
        uint8_t planeIndex; 
        if (extents[i].intersect(precomputedNumerator, precomputedDenominator, tn, tf, planeIndex)) { 
            if (tn < tNear) { 
                intersectedMesh = meshes[i].get(); 
                tNear = tn; 
                // normal = planeSetNormals[planeIndex];
            } 
        } 
    } 
    */ 
 
    uint8_t planeIndex; 
    float tNear = 0, tFar = kInfinity; // tNear, tFar for the intersected extents 
    if (!octree->root->nodeExtents.intersect(precomputedNumerator, precomputedDenominator, tNear, tFar, planeIndex) || tFar < 0) 
        return false; 
    tHit = tFar; 
    std::priority_queue<BVH::Octree::QueueElement> queue; 
    queue.push(BVH::Octree::QueueElement(octree->root, 0)); 
    while (!queue.empty() && queue.top().t < tHit) { 
        const Octree::OctreeNode *node = queue.top().node; 
        queue.pop(); 
        if (node->isLeaf) { 
            for (const auto& e: node->nodeExtentsList) {
                float t = kInfinity; 
                float temp0 = kInfinity, temp1 = kInfinity;
                Vector3 vec3orig = Vector3(orig.x, orig.y, orig.z);
                Vector3 vec3dir = Vector3(dir.x, dir.y, dir.z);
                if (e->mesh->intersect(vec3orig, vec3dir, temp0, temp1) && t < tHit) { 
                    tHit = t; 
                    intersectedMesh = e->mesh; 
                } 
            } 
        } 
        else { 
            for (uint8_t i = 0; i < 8; ++i) { 
                if (node->child[i] != nullptr) { 
                    float tNearChild = 0, tFarChild = tFar; 
                    if (node->child[i]->nodeExtents.intersect(precomputedNumerator, precomputedDenominator, tNearChild, tFarChild, planeIndex)) { 
                        float t = (tNearChild < 0 && tFarChild >= 0) ? tFarChild : tNearChild; 
                        queue.push(BVH::Octree::QueueElement(node->child[i], t)); 
                    } 
                } 
            } 
        } 
    } 
 
    return (intersectedMesh != nullptr); 
}


const Vec3f BVH::planeSetNormals[BVH::kNumPlaneSetNormals] = { 
    Vec3f(1, 0, 0), 
    Vec3f(0, 1, 0), 
    Vec3f(0, 0, 1), 
    Vec3f( sqrtf(3) / 3.f,  sqrtf(3) / 3.f, sqrtf(3) / 3.f), 
    Vec3f(-sqrtf(3) / 3.f,  sqrtf(3) / 3.f, sqrtf(3) / 3.f), 
    Vec3f(-sqrtf(3) / 3.f, -sqrtf(3) / 3.f, sqrtf(3) / 3.f), 
    Vec3f( sqrtf(3) / 3.f, -sqrtf(3) / 3.f, sqrtf(3) / 3.f) 
}; 
 
BVH::BVH(std::vector<std::unique_ptr<const Mesh>>& m) : AccelerationStructure(m) 
{ 
    Extents sceneExtents; // that's the extent of the entire scene which we need to compute for the octree 
    extentsList.reserve(meshes.size()); 
    for (uint32_t i = 0; i < meshes.size(); ++i) { 
        for (uint8_t j = 0; j < kNumPlaneSetNormals; ++j) { 
            for (const auto vtx : meshes[i]->vertexPool) { 
                float d = dot(planeSetNormals[j], vtx); 
                // set dNEar and dFar
                if (d < extentsList[i].d[j][0]) extentsList[i].d[j][0] = d; 
                if (d > extentsList[i].d[j][1]) extentsList[i].d[j][1] = d; 
            } 
        } 
        sceneExtents.extendBy(extentsList[i]); // expand the scene extent of this object's extent 
        extentsList[i].mesh = meshes[i].get(); // the extent itself needs to keep a pointer to the object its holds 
    } 
 
    // Now that we have the extent of the scene we can start building our octree
    // Using C++ make_unique function here but you don't need to, just to learn something... 
    octree = new Octree(sceneExtents); 
 
    for (uint32_t i = 0; i < meshes.size(); ++i) { 
        octree->insert(&extentsList[i]); 
    } 
 
    // Build from bottom up
    octree->build(); 
} 
 
bool BVH::Extents::intersect( 
    const float* precomputedNumerator, 
    const float* precomputedDenominator, 
    float& tNear,   // tn and tf in this method need to be contained 
    float& tFar,    // within the range [tNear:tFar] 
    uint8_t& planeIndex) const 
{ 
    for (uint8_t i = 0; i < kNumPlaneSetNormals; ++i) { 
        float tNearExtents = (d[i][0] - precomputedNumerator[i]) / precomputedDenominator[i]; 
        float tFarExtents = (d[i][1] - precomputedNumerator[i]) / precomputedDenominator[i]; 
        if (precomputedDenominator[i] < 0) std::swap(tNearExtents, tFarExtents); 
        if (tNearExtents > tNear) tNear = tNearExtents, planeIndex = i; 
        if (tFarExtents < tFar) tFar = tFarExtents; 
        if (tNear > tFar) return false; 
    } 
 
    return true; 
} 