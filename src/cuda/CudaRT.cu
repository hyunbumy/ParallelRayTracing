#include <fstream>
#include <sstream>
#include <random>

#include "CudaRT.h"
#include "CudaSphere.h"
#include "CudaTriangle.h"

__device__
float3 Trace(float3& rayorig, float3& raydir, CudaObject** objects, 
             int objSize, int depth)
{
    //if (raydir.length() != 1) std::cerr << "Error " << raydir << std::endl;
    // find intersection of this ray with the sphere in the scene
    float tnear = INFINITY;
    const CudaObject* object = GetClosestObject(rayorig, raydir, tnear,
                                                objects, objSize);
    
    // if there's no intersection return black or background color
    if (object == nullptr) return make_float3(2, 2, 2);
    float3 surfaceColor = make_float3(0,0,0); // color of the ray/surfaceof the object intersected by the ray
    float3 phit = rayorig + raydir * tnear; // point of intersection
    float3 nhit = object->CalculateHit(phit); // normal at the intersection point
    nhit = normalize(nhit); // normalize normal direction
    // If the normal and the view direction are not opposite to each other
    // reverse the normal direction. That also means we are inside the sphere so set
    // the inside bool to true. Finally reverse the sign of IdotN which we want
    // positive.
    float bias = 1e-4; // add some bias to the point from which we will be tracing
    bool inside = false;
    if (dot(raydir, nhit) > 0)
    {
        nhit *= -1;
        inside = true;
    }

    if ((object->transparency > 0 || object->reflection > 0) && depth < MAX_RAY_DEPTH) {
        float facingratio = -1 * dot(raydir, nhit);
        // change the mix value to tweak the effect
        float fresneleffect = Mix(pow(1 - facingratio, 3), 1, 0.1);
        // compute reflection direction (not need to normalize because all vectors
        // are already normalized)
        float3 refldir = raydir - nhit * 2 * dot(raydir, nhit);
        refldir = normalize(refldir);
        auto reflOrig = phit + nhit * bias;
        float3 reflection = Trace(reflOrig, refldir, objects, objSize, depth + 1);
        float3 refraction = make_float3(0,0,0);
        // if the sphere is also transparent compute refraction ray (transmission)
        if (object->transparency) {
            float ior = 1.1, eta = (inside) ? ior : 1 / ior; // are we inside or outside the surface?
            float cosi = -1 * dot(nhit, raydir);
            float k = 1 - eta * eta * (1 - cosi * cosi);
            float3 refrdir = raydir * eta + nhit * (eta *  cosi - sqrtf(k));
            refrdir = normalize(refrdir);
            reflOrig = phit - nhit * bias;
            refraction = Trace(reflOrig, refrdir, objects, objSize, depth + 1);
        }
        // the result is a mix of reflection and refraction (if the sphere is transparent)
        surfaceColor = (
            reflection * fresneleffect +
            refraction * (1 - fresneleffect) * object->transparency) * object->surfaceColor;
    }
    else {
        // it's a diffuse object, no need to raytrace any further
        for (unsigned i = 0; i < objSize; ++i) {
            if (objects[i]->emissionColor.x > 0) {
                // this is a light
                float3 transmission = make_float3(1,1,1);
                float3 lightDirection = -1 * objects[i]->CalculateHit(phit);
                lightDirection = normalize(lightDirection);
                for (unsigned j = 0; j < objSize; ++j) {
                    if (i != j) {
                        float t0, t1;
                        auto orig = phit + nhit * bias;
                        if (objects[j]->Intersect(orig, lightDirection, t0, t1)) {
                            transmission = make_float3(0,0,0);
                            break;
                        }
                    }
                }
                surfaceColor += object->surfaceColor * transmission *
                fmaxf(float(0), dot(nhit, lightDirection)) * objects[i]->emissionColor;
            }
        }
    }
    
    return surfaceColor + object->emissionColor;
}

__global__
void Render(float3* image, CudaObject** objects, int objectSize,
            unsigned int width, unsigned int height)
{
    float invWidth = 1 / float(width), invHeight = 1 / float(height);
    float fov = 30, aspectRatio = width / float(height);
    float angle = tanf(M_PI * 0.5 * fov / 180.);

    // Parallelization
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;   
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int i = x + width * y;

    // Trace rays
    float xx = (2 * ((x + 0.5f) * invWidth) - 1) * angle * aspectRatio; 
    float yy = (1 - 2 * ((y + 0.5f) * invHeight)) * angle; 
    float3 raydir = make_float3(xx, yy, -1);
    raydir -= make_float3(0,0,0);   // Cam direction
    raydir = normalize(raydir);
    float3 position = make_float3(0,6,30);
    image[i] = Trace(position, raydir, objects, objectSize, 0);
}

__global__
void Allocate(CudaObject** objects, int size, float3* objectInfo)
{
    // LcLd
    // new (objects[0]) CudaSphere(make_float3(0, -10004, -20), 10000, make_float3(0.2, 0.2, 0.2), 0, 0);
    // new (objects[1]) CudaSphere(make_float3(0, 0, -20), 4.0, make_float3(1.0, 0.32, 0.36), 1, 0.5);
    // new (objects[2]) CudaSphere(make_float3(5, -1, -15), 2, make_float3(0.9, 0.76, 0.46), 1, 0.0);
    // new (objects[3]) CudaSphere(make_float3(5, 0, -25), 3, make_float3(0.65, 0.77, 0.97), 1, 0.0);
    // new (objects[4]) CudaSphere(make_float3(-5.5, 0, -15), 3, make_float3(0.9, 0.9, 0.9), 1, 0.0);
    // new (objects[size-1]) CudaSphere(make_float3(0, 20, -30), 3, make_float3(0, 0, 0), 0, 0.0f, make_float3(3,3,3));
    
    // HcLd
    // for (int i = 0; i < size-1; ++i)
    // {
    //     new (objects[i]) CudaTriangle(objectInfo[i*3], objectInfo[i*3+1], objectInfo[i*3+2], make_float3(1, 1, 0.25), 1, 0);
    // }
    // new (objects[size-1]) CudaSphere(make_float3(0, 0.5, 0.3), 3, make_float3(0, 0, 0), 0, 0.0f, make_float3(7,7,7));

    // LcHd
    // for (int i = 0; i < size - 1; ++i)
    // {
    //     float3 info = objectInfo[i*3+2];
    //     new (objects[i]) CudaSphere(objectInfo[i*3], info.x, objectInfo[i*3+1], info.y, info.z);
    // }
    // new (objects[size-1]) CudaSphere(make_float3(0, 20, -30), 3, make_float3(0, 0, 0), 0, 0.0f, make_float3(3,3,3));

    // HcHd
    for (int i = 0; i < size-1; ++i)
    {
        new (objects[i]) CudaTriangle(objectInfo[i*3], objectInfo[i*3+1], objectInfo[i*3+2], make_float3(1, 0.32, 0.36), 1, 0);
    }
    new (objects[size-1]) CudaSphere(make_float3(0, 0, 0), 2, make_float3(0, 0, 0), 0, 0.0f, make_float3(3,3,3));
}

void CudaRT::RenderWrapper(float3* image, unsigned width, unsigned height)
{
    // dim3 is CUDA specific type, block and grid are required to schedule CUDA threads over streaming multiprocessors
    dim3 block(8, 8, 1);   
    dim3 grid(width / block.x, height / block.y, 1);

    float3* output;    // pointer to memory for image on the device (GPU VRAM)
    checkCudaErrors(cudaMalloc(&output, width*height*sizeof(float3)));

    // Parse TriMesh file
    float3* objectInfo;
    int infoSize;
    std::vector<std::string> files{
        "../../mesh_files/cow.geo", "../../mesh_files/cow2.geo",
        "../../mesh_files/cow3.geo", "../../mesh_files/cow4.geo",
        "../../mesh_files/cow5.geo"
    };
    infoSize = ParseTriMesh(files, objectInfo);

    // Create spheres
    // checkCudaErrors(cudaMallocManaged(&objectInfo, 3*100*sizeof(CudaSphere)));
    // int seed = 1362;//time(NULL) % 10000; //2057;
    // srand(seed);
    // for(int i = 0; i < 100; i++) {
    //     float x = ( ((float) rand()) / RAND_MAX ) * 20 - 10; //ranges from -10 to 10
    //     float y = ( ((float) rand()) / RAND_MAX ) * 20 - 10;      //ranges from -2.5 to 2.5
    //     float z = ( ((float) rand()) / RAND_MAX ) * 100 * (-1) - 20; //ranges from -40 to -10

    //     float radius = ( ((float) rand()) / RAND_MAX ) * 2.5; //ranges from 0 to 3

    //     float r = ( ((float) rand()) / RAND_MAX ) * 0.6 + 0.4;
    //     float g = ( ((float) rand()) / RAND_MAX ) * 0.6 + 0.4;
    //     float b = ( ((float) rand()) / RAND_MAX ) * 0.6 + 0.4;

    //     objectInfo[i*3] = make_float3(x, y, z);    // Center
    //     objectInfo[i*3+1] = make_float3(r,g,b);    // Color
    //     objectInfo[i*3+2] = make_float3(radius, 1, 0.5);   // Radius, reflectivity, transparent
    // }
    // infoSize = 100;

    // Scence creation
    CudaObject** objects;
    int size = infoSize + 1;
    CudaObject** objects_h = (CudaObject**)malloc(size*sizeof(CudaObject*));
    checkCudaErrors(cudaMalloc((void**)&objects, size*sizeof(CudaObject*)));
    for (int i = 0; i < size-1; ++i)
    {
        checkCudaErrors(cudaMalloc((void**)&objects_h[i], sizeof(CudaTriangle)));
    }
    checkCudaErrors(cudaMalloc((void**)&objects_h[size-1], sizeof(CudaSphere)));
    checkCudaErrors(cudaMemcpy(objects, objects_h, size*sizeof(CudaObject*), cudaMemcpyHostToDevice));
    Allocate<<<1,1>>>(objects,size, objectInfo);

    std::cout << "Memory allocated" << std::endl;
    
    // schedule threads on device and launch CUDA kernel from host
    Render<<< grid, block >>>(output, objects, size, width, height);
    
    // Single GPU Thread
    // Render<<< 1, 1 >>>(output, spheres, size, width, height);

    // Wait to synchronize
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "Finish synchronization" << std::endl;

    // Copy results back
    cudaMemcpy(image, output, width*height*sizeof(float3), cudaMemcpyDeviceToHost);

    // free CUDA memory
    checkCudaErrors(cudaFree(output));
    for (int i = 0; i < size; ++i)
    {
        checkCudaErrors(cudaFree(objects_h[i]));
    }
    checkCudaErrors(cudaFree(objects));
    checkCudaErrors(cudaFree(objectInfo));
    free(objects_h);
}

int CudaRT::ParseTriMesh(std::vector<std::string> files, float3*& triangles)
{
    std::vector<float3> tempTriangles;
    for (int i = 0; i < files.size(); ++i)
    {
        std::string filename(files[i]);
        std::ifstream ifs(filename);
        std::stringstream ss;
        ss << ifs.rdbuf();

        // Parse vertices
        if (filename.find(".geo") != std::string::npos)
        {
            std::vector<float3> vertices;
            uint32_t numVertices;
            ss >> numVertices;

            for(uint32_t i = 0; i < numVertices; i++) {
                float3 temp;
                ss >> temp.x;
                ss >> temp.y;
                ss >> temp.z;
                vertices.push_back(temp);
            }

            // Parse triangles
            uint32_t numTriangles;
            ss >> numTriangles;

            for(uint32_t i = 0; i  < numTriangles; i++) {
                int v0, v1, v2;
                ss >> v0;
                ss >> v1;
                ss >> v2;
                tempTriangles.emplace_back(vertices[v0]);
                tempTriangles.emplace_back(vertices[v1]);
                tempTriangles.emplace_back(vertices[v2]);
            }
        }
        else if (filename.find(".dae") != std::string::npos)
        {
            uint32_t numVertices;
            ss >> numVertices;

            std::vector<float3> vertices;
            for(uint32_t i = 0; i < numVertices; i++) {
                float3 temp;
                ss >> temp.x;
                ss >> temp.y;
                ss >> temp.z;
                //std::cout << temp.x << " " << temp.y << " " << temp.z << std::endl;
                vertices.push_back(temp);
            }

            uint32_t numTriangles;
            ss >> numTriangles;

            // Allocate triangle information
            checkCudaErrors(cudaMallocManaged((void**)&triangles, 3*numTriangles*sizeof(float3)));

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
                tempTriangles.emplace_back(vertices[v0]);
                tempTriangles.emplace_back(vertices[v1]);
                tempTriangles.emplace_back(vertices[v2]);
            }
        }
        ifs.close();
    }

    // Allocate triangle information
    checkCudaErrors(cudaMallocManaged((void**)&triangles, tempTriangles.size()*sizeof(float3)));
    for (int i = 0; i < tempTriangles.size(); ++i)
    {
        triangles[i] = tempTriangles[i];
    }
    
    return tempTriangles.size();
}
