#include "CudaRT.h"

__device__
float3 Trace(float3& rayorig, float3& raydir, CudaSphere* objects, 
             int objSize, int depth)
{
    //if (raydir.length() != 1) std::cerr << "Error " << raydir << std::endl;
    // find intersection of this ray with the sphere in the scene
    float tnear = INFINITY;
    const CudaSphere* object = GetClosestObject(rayorig, raydir, tnear,
                                                objects, objSize);
    
    // if there's no intersection return black or background color
    if (object == nullptr) return make_float3(2, 2, 2);
    float3 surfaceColor = make_float3(0,0,0); // color of the ray/surfaceof the object intersected by the ray
    float3 phit = rayorig + raydir * tnear; // point of intersection
    float3 nhit = phit - object->center; // normal at the intersection point
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
            if (objects[i].emissionColor.x > 0) {
                // this is a light
                float3 transmission = make_float3(1,1,1);
                float3 lightDirection = -1 * (phit - objects[i].center);
                lightDirection = normalize(lightDirection);
                for (unsigned j = 0; j < objSize; ++j) {
                    if (i != j) {
                        float t0, t1;
                        auto orig = phit + nhit * bias;
                        if (objects[j].Intersect(orig, lightDirection, t0, t1)) {
                            transmission = make_float3(0,0,0);
                            break;
                        }
                    }
                }
                surfaceColor += object->surfaceColor * transmission *
                fmaxf(float(0), dot(nhit, lightDirection)) * objects[i].emissionColor;
            }
        }
    }
    
    return surfaceColor + object->emissionColor;
}

__global__
void Render(float3* image, CudaSphere* objects, int objectSize,
            unsigned int width, unsigned int height)
{
    float invWidth = 1 / float(width), invHeight = 1 / float(height);
    float fov = 30, aspectRatio = width / float(height);
    float angle = tanf(M_PI * 0.5 * fov / 180.);

    // // Single GPU Thread
    // float3* pixel = image;
    // for (unsigned y = 0; y < height; ++y) {
    //     for (unsigned x = 0; x < width; ++x, ++pixel) {
    //         float xx = (2 * ((x + 0.5) * invWidth) - 1) * angle * aspectRatio;
    //         float yy = (1 - 2 * ((y + 0.5) * invHeight)) * angle;
    //         float3 raydir = make_float3(xx, yy, -1);
    //         raydir = normalize(raydir);
    //         float3 zero = make_float3(0,0,0);
    //         *pixel = Trace(zero, raydir, objects, objectSize, 0);
    //     }
    // }

    // Parallelization
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;   
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int i = x + width * y;

    // Trace rays
    float xx = (2 * ((x + 0.5f) * invWidth) - 1) * angle * aspectRatio; 
    float yy = (1 - 2 * ((y + 0.5f) * invHeight)) * angle; 
    float3 raydir = make_float3(xx, yy, -1); 
    raydir = normalize(raydir);
    float3 zero = make_float3(0,0,0);
    image[i] = Trace(zero, raydir, objects, objectSize, 0);
}

void CudaRT::RenderWrapper(float3* image, unsigned width, unsigned height)
{
    float3* output;    // pointer to memory for image on the device (GPU VRAM)
    checkCudaErrors(cudaMalloc(&output, width*height*sizeof(float3)));

    CudaSphere* spheres;
    int size = 6;
    checkCudaErrors(cudaMallocManaged(&spheres, size*sizeof(CudaSphere)));
    spheres[0] = CudaSphere(make_float3(0, -10004, -20), 10000, make_float3(0.2, 0.2, 0.2), 0, 0);
    spheres[1] = CudaSphere(make_float3(0, 0, -20), 4.0, make_float3(1.0, 0.32, 0.36), 1, 0.5);
    spheres[2] = CudaSphere(make_float3(5, -1, -15), 2, make_float3(0.9, 0.76, 0.46), 1, 0.0);
    spheres[3] = CudaSphere(make_float3(5, 0, -25), 3, make_float3(0.65, 0.77, 0.97), 1, 0.0);
    spheres[4] = CudaSphere(make_float3(-5.5, 0, -15), 3, make_float3(0.9, 0.9, 0.9), 1, 0.0);
    spheres[5] = CudaSphere(make_float3(0.0, 20, -30), 3, make_float3(0, 0, 0), 0, 0.0f, make_float3(3,3,3));

    std::cout << "Memory allocated" << std::endl;
            
    // dim3 is CUDA specific type, block and grid are required to schedule CUDA threads over streaming multiprocessors
    dim3 block(8, 8, 1);   
    dim3 grid(width / block.x, height / block.y, 1);
    
    // schedule threads on device and launch CUDA kernel from host
    Render<<< grid, block >>>(output, spheres, size, width, height);
    
    // Single GPU Thread
    // Render<<< 1, 1 >>>(output, spheres, size, width, height);

    // Wait to synchronize
    checkCudaErrors(cudaDeviceSynchronize());
    std::cout << "Finish synchronization" << std::endl;

    // Copy results back
    cudaMemcpy(image, output, width*height*sizeof(float3), cudaMemcpyDeviceToHost);

    // free CUDA memory
    checkCudaErrors(cudaFree(output));
    checkCudaErrors(cudaFree(spheres));
}
