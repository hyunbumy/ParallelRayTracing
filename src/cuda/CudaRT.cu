#include <cmath>
#include <iostream>

#include "CudaRT.h"
#include "CudaSphere.h"
#include "helper_math.h"

__device__
float3 Trace(float3& rayorig, float3& raydir, CudaSphere* spheres, int depth)
{
    return make_float3(255, 255, 255);
}

__global__
void Render(float3* image, CudaSphere* spheres,
            unsigned int width, unsigned int height)
{
    float3* pixel = image;
    float invWidth = 1 / float(width), invHeight = 1 / float(height);
    float fov = 30.0f, aspectRatio = width / float(height);
    float angle = tanf(M_PI * 0.5f * fov / 180.0f);

    // Trace rays
    for (unsigned y = 0; y < height; ++y) { 
        for (unsigned x = 0; x < width; ++x, ++pixel) { 
            float xx = (2 * ((x + 0.5f) * invWidth) - 1) * angle * aspectRatio; 
            float yy = (1 - 2 * ((y + 0.5f) * invHeight)) * angle; 
            float3 raydir = make_float3(xx, yy, -1); 
            normalize(raydir);
            float3 zero = make_float3(0,0,0);
            *pixel = Trace(zero, raydir, spheres, 0); 
        } 
    }
}

void CudaRT::RenderWrapper(float* image, unsigned width, unsigned height)
{
    float3* output;    // pointer to memory for image on the device (GPU VRAM)
    cudaMallocManaged(&output, width*height*sizeof(float3));

    CudaSphere* spheres;
    cudaMallocManaged(&spheres, sizeof(CudaSphere));
    spheres[0] = CudaSphere(make_float3(0, 0, -20), 4.0f, make_float3(1.0f, 0.32f, 0.36f), 1, 0);

    std::cout << "Memory allocated" << std::endl;
            
    // dim3 is CUDA specific type, block and grid are required to schedule CUDA threads over streaming multiprocessors
    dim3 block(8, 8, 1);   
    dim3 grid(width / block.x, height / block.y, 1);
    
    // schedule threads on device and launch CUDA kernel from host
    Render<<< grid, block >>>(output, spheres, width, height);

    // Wait to synchronize
    cudaDeviceSynchronize();
    std::cout << "Finish synchronization" << std::endl;

    // Copy results back
    for (int i = 0; i < width*height; ++i)
    {
        image[i*3] = output[i].x;
        image[i*3+1] = output[i].y;
        image[i*3+2] = output[i].z;
    }

    // free CUDA memory
    cudaFree(output);
    cudaFree(spheres);
}
