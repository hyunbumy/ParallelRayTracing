#pragma once

#include <iostream>

#include "CudaSphere.h"
#include "helper_math.h"

#define MAX_RAY_DEPTH 5

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

__inline__ void check_cuda(cudaError_t result, char const *const func, 
                           const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__
__inline__ static CudaSphere* GetClosestObject(float3& rayorig, float3& raydir,
                                               float& tnear,
                                               CudaSphere* objects, int objSize)
{
    CudaSphere* object = nullptr;
    for (unsigned i = 0; i < objSize; ++i) {
        float t0 = INFINITY, t1 = INFINITY;
        if (objects[i].Intersect(rayorig, raydir, t0, t1)) {
            if (t0 < 0) t0 = t1;
            if (t0 < tnear) {
                tnear = t0;
                object = &objects[i];
            }
        }
    }
    return object;
}
