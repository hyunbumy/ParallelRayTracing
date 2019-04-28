#include "accelerator.h"
#include "AccelerationStructure.h"
class BVH : public AccelerationStructure 
{ 
    static const uint8_t kNumPlaneSetNormals = 7; 
    static const Vec3f planeSetNormals[kNumPlaneSetNormals]; 
    struct Extents 
    { 
        Extents() 
        { 
            for (uint8_t i = 0;  i < kNumPlaneSetNormals; ++i) 
                d[i][0] = kInfinity, d[i][1] = -kInfinity; 
        } 
        void extendBy(const Extents& e) 
        { 
 
            for (uint8_t i = 0;  i < kNumPlaneSetNormals; ++i) { 
                if (e.d[i][0] < d[i][0]) d[i][0] = e.d[i][0]; 
                if (e.d[i][1] > d[i][1]) d[i][1] = e.d[i][1]; 
            } 
        } 
        /* inline */ 
        Vec3f centroid() const 
        { 
            return Vec3f( 
                d[0][0] + d[0][1] * 0.5, 
                d[1][0] + d[1][1] * 0.5, 
                d[2][0] + d[2][1] * 0.5); 
        } 
        bool intersect(const float*, const float*, float&, float&, uint8_t&) const; 
        float d[kNumPlaneSetNormals][2]; 
        const Mesh* mesh; 
    }; 
 
    struct Octree 
    { 
        Octree(const Extents& sceneExtents) 
        { 
            float xDiff = sceneExtents.d[0][1] - sceneExtents.d[0][0]; 
            float yDiff = sceneExtents.d[1][1] - sceneExtents.d[1][0]; 
            float zDiff = sceneExtents.d[2][1] - sceneExtents.d[2][0]; 
            float maxDiff = std::max(xDiff, std::max(yDiff, zDiff)); 
            Vec3f minPlusMax( 
                sceneExtents.d[0][0] + sceneExtents.d[0][1], 
                sceneExtents.d[1][0] + sceneExtents.d[1][1], 
                sceneExtents.d[2][0] + sceneExtents.d[2][1]); 
            bbox[0] = (minPlusMax - maxDiff) * 0.5; 
            bbox[1] = (minPlusMax + maxDiff) * 0.5; 
            root = new OctreeNode; 
        } 
 
        ~Octree() { deleteOctreeNode(root); } 
 
        void insert(const Extents* extents) { insert(root, extents, bbox, 0); } 
        void build() { build(root, bbox); }; 
 
        struct OctreeNode 
        { 
            OctreeNode* child[8] = { nullptr }; 
            std::vector<const Extents *> nodeExtentsList; // pointer to the objects extents 
            Extents nodeExtents; // extents of the octree node itself 
            bool isLeaf = true; 
        }; 
 
        struct QueueElement 
        { 
            const OctreeNode *node; // octree node held by this element in the queue 
            float t; // distance from the ray origin to the extents of the node 
            QueueElement(const OctreeNode *n, float tn) : node(n), t(tn) {} 
            // priority_queue behaves like a min-heap
            friend bool operator < (const QueueElement &a, const QueueElement &b) { return a.t > b.t; } 
        }; 
 
        OctreeNode* root = nullptr; // make unique son don't have to manage deallocation 
        BBox<> bbox; 
 
    private: 
 
        void deleteOctreeNode(OctreeNode*& node) 
        { 
            for (uint8_t i = 0; i < 8; i++) { 
                if (node->child[i] != nullptr) { 
                    deleteOctreeNode(node->child[i]); 
                } 
            } 
            delete node; 
        } 
 
        void insert(OctreeNode*& node, const Extents* extents, const BBox<>& bbox, uint32_t depth) 
        { 
            if (node->isLeaf) { 
                if (node->nodeExtentsList.size() == 0 || depth == 16) { 
                    node->nodeExtentsList.push_back(extents); 
                } 
                else { 
                    node->isLeaf = false; 
                    // Re-insert extents held by this node
                    while (node->nodeExtentsList.size()) { 
                        insert(node, node->nodeExtentsList.back(), bbox, depth); 
                        node->nodeExtentsList.pop_back(); 
                    } 
                    // Insert new extent
                    insert(node, extents, bbox, depth); 
                } 
            } 
            else { 
                // Need to compute in which child of the current node this extents should
                // be inserted into
                Vec3f extentsCentroid = extents->centroid(); 
                Vec3f nodeCentroid = (bbox[0] + bbox[1]) * 0.5; 
                BBox<> childBBox; 
                uint8_t childIndex = 0; 
                // x-axis
                if (extentsCentroid.x > nodeCentroid.x) { 
                    childIndex = 4; 
                    childBBox[0].x = nodeCentroid.x; 
                    childBBox[1].x = bbox[1].x; 
                } 
                else { 
                    childBBox[0].x = bbox[0].x; 
                    childBBox[1].x = nodeCentroid.x; 
                } 
                // y-axis
                if (extentsCentroid.y > nodeCentroid.y) { 
                    childIndex += 2; 
                    childBBox[0].y = nodeCentroid.y; 
                    childBBox[1].y = bbox[1].y; 
                } 
                else { 
                    childBBox[0].y = bbox[0].y; 
                    childBBox[1].y = nodeCentroid.y; 
                } 
                // z-axis
                if (extentsCentroid.z > nodeCentroid.z) { 
                    childIndex += 1; 
                    childBBox[0].z = nodeCentroid.z; 
                    childBBox[1].z = bbox[1].z; 
                } 
                else { 
                    childBBox[0].z = bbox[0].z; 
                    childBBox[1].z = nodeCentroid.z; 
                } 
 
                // Create the child node if it doesn't exsit yet and then insert the extents in it
                if (node->child[childIndex] == nullptr) 
                    node->child[childIndex] = new OctreeNode; 
                insert(node->child[childIndex], extents, childBBox, depth + 1); 
            } 
        } 
 
        void build(OctreeNode*& node, const BBox<>& bbox) 
        { 
            if (node->isLeaf) { 
                for (const auto& e: node->nodeExtentsList) { 
                    node->nodeExtents.extendBy(*e); 
                } 
            } 
            else { 
                for (uint8_t i = 0; i < 8; ++i) { 
                        if (node->child[i]) { 
                        BBox<> childBBox; 
                        Vec3f centroid = bbox.centroid(); 
                        // x-axis
                        childBBox[0].x = (i & 4) ? centroid.x : bbox[0].x; 
                        childBBox[1].x = (i & 4) ? bbox[1].x : centroid.x; 
                        // y-axis
                        childBBox[0].y = (i & 2) ? centroid.y : bbox[0].y; 
                        childBBox[1].y = (i & 2) ? bbox[1].y : centroid.y; 
                        // z-axis
                        childBBox[0].z = (i & 1) ? centroid.z : bbox[0].z; 
                        childBBox[1].z = (i & 1) ? bbox[1].z : centroid.z; 
 
                        // Inspect child
                        build(node->child[i], childBBox); 
 
                        // Expand extents with extents of child
                        node->nodeExtents.extendBy(node->child[i]->nodeExtents); 
                    } 
                } 
            } 
        } 
    }; 
 
    std::vector<Extents> extentsList; 
    Octree* octree = nullptr; 
public: 
    BVH(std::vector<std::unique_ptr<const Mesh>>& m); 
    bool intersect(const Vec3f&, const Vec3f&, const uint32_t&, float&) const; 
    ~BVH() { delete octree; } 
}; 