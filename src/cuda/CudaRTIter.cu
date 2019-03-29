#include "CudaRTIter.h"

__device__
unsigned GetCurrentIndex(unsigned width, unsigned height)
{
    unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
    return x + width * y;
}

__device__
float3 Trace(float3& direction, float3& base, unsigned depth,
             CudaSphere* objects, int objSize)
{
    // Ray Tracing finished
    if (direction.x == INFINITY && direction.y == INFINITY && direction.z == INFINITY)
    {
        return make_float3(INFINITY,INFINITY,INFINITY);
    }

    // Get Intersection
    float tnear = INFINITY;
    const CudaSphere* object = GetClosestObject(
        base, direction, tnear, objects, objSize
    );
    if (object == nullptr)
    {
        direction = make_float3(INFINITY, INFINITY, INFINITY);
        return make_float3(2,2,2);
    }

    float3 phit = base + direction * tnear;
    float3 nhit = phit - object->center;
    nhit = normalize(nhit);
    float bias = 1e-4;
    bool inside = false;
    if (dot(direction, nhit) > 0)
    {
        nhit *= -1;
        inside = true;
    }
    float3 intersection = phit + nhit * bias;
    auto surfaceColor = make_float3(0,0,0);

    // Check reflection
    if ((object->transparency > 0 || object->reflection > 0) && depth < MAX_RAY_DEPTH-1)
    {
        float facingRatio = -1 * dot(direction, nhit);
        // change the mix value to tweak the effect
        float fresnelEffect = Mix(pow(1 - facingRatio, 3), 1, 0.1);
        // compute reflection direction
        float3 reflDir = direction - nhit * 2 * dot(direction, nhit);

        // Update next ray
        base = intersection;
        direction = normalize(reflDir);
        return object->surfaceColor * fresnelEffect;
    }

    // Diffuse object, compute illumination
    for (unsigned i = 0; i < objSize; ++i)
    {
        if (objects[i].emissionColor.x > 0)
        {
            // this is a light
            float3 transmission = make_float3(1,1,1);
            float3 lightDirection = -1 * (phit - objects[i].center);
            lightDirection = normalize(lightDirection);

            // Check for obstruction
            for (unsigned j = 0; j < objSize; ++j) {
                if (i != j) {
                    float t0, t1;
                    if (objects[j].Intersect(intersection, lightDirection, t0, t1))
                    {
                        transmission = make_float3(0,0,0);
                        break;
                    }
                }
            }
            surfaceColor += object->surfaceColor * transmission *
                            fmaxf(float(0), dot(nhit, lightDirection)) *
                            objects[i].emissionColor;
        }
    }

    // If diffuse object, stop ray tracing
    direction = make_float3(INFINITY, INFINITY, INFINITY);
    return surfaceColor + object->emissionColor;
}

__global__
void Initialize(float3* directions, float3* bases, 
                unsigned width, unsigned height)
{
    unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned i = GetCurrentIndex(width, height);

    // Directions
    float invWidth = 1 / float(width), invHeight = 1 / float(height);
    float fov = 30, aspectRatio = width / float(height);
    float angle = tanf(M_PI * 0.5 * fov / 180.);

    float xx = (2 * ((x + 0.5f) * invWidth) - 1) * angle * aspectRatio; 
    float yy = (1 - 2 * ((y + 0.5f) * invHeight)) * angle; 
    float3 rayDir = normalize(make_float3(xx, yy, -1));
    directions[i] = rayDir;

    // Bases
    bases[i] = make_float3(0,0,0);
}

__global__
void Render(float3* layers, unsigned depth, float3* directions, float3* bases, 
            CudaSphere* objects, int objSize, unsigned width, unsigned height)
{
    unsigned i = GetCurrentIndex(width, height);

    layers[i+width*height*depth] = Trace(
        directions[i], bases[i], depth, objects, objSize
    );
}

__global__
void Reassemble(float3* layers, unsigned depth, 
                unsigned width, unsigned height)
{
    unsigned i = GetCurrentIndex(width, height);
    float3 top = layers[i+width*height*depth];
    
    if (top.x != INFINITY)
    {
        layers[i+width*height*(depth-1)] = top*layers[i+width*height*(depth-1)];
    }
}

CudaRTIter::CudaRTIter(unsigned width, unsigned height)
    :mWidth(width)
    ,mHeight(height)
{
    checkCudaErrors(
        cudaMalloc(&mLayers, MAX_RAY_DEPTH*width*height*sizeof(float3))
    );

    checkCudaErrors(
        cudaMalloc(&mDirections, width*height*sizeof(float3))
    );

    checkCudaErrors(
        cudaMalloc(&mBases, width*height*sizeof(float3))
    );
}

CudaRTIter::~CudaRTIter()
{
    // free CUDA memory
    checkCudaErrors(cudaFree(mLayers));
    checkCudaErrors(cudaFree(mDirections));
    checkCudaErrors(cudaFree(mBases));
}

void CudaRTIter::RenderWrapper(float3* image)
{
    CudaSphere* spheres;
    int size = 6;
    checkCudaErrors(cudaMallocManaged(&spheres, size*sizeof(CudaSphere)));
    spheres[0] = CudaSphere(make_float3(0, -10004, -20), 10000, make_float3(0.2, 0.2, 0.2), 0, 0);
    spheres[1] = CudaSphere(make_float3(0, 0, -20), 4.0, make_float3(1.0, 0.32, 0.36), 1, 0.5);
    spheres[2] = CudaSphere(make_float3(5, -1, -15), 2, make_float3(0.9, 0.76, 0.46), 1, 0.0);
    spheres[3] = CudaSphere(make_float3(5, 0, -25), 3, make_float3(0.65, 0.77, 0.97), 1, 0.0);
    spheres[4] = CudaSphere(make_float3(-5.5, 0, -15), 3, make_float3(0.9, 0.9, 0.9), 1, 0.0);
    spheres[5] = CudaSphere(make_float3(0.0, 20, -30), 3, make_float3(0, 0, 0), 0, 0.0f, make_float3(3,3,3));

    // dim3 is CUDA specific type, block and grid are required to schedule CUDA threads over streaming multiprocessors
    dim3 block(8, 8, 1);   
    dim3 grid(mWidth / block.x, mHeight / block.y, 1);

    std::cout << "Initialize" << std::endl;
    // schedule threads on device and launch CUDA kernel from host
    Initialize<<< grid, block >>>(mDirections, mBases, mWidth, mHeight);
    checkCudaErrors(cudaDeviceSynchronize());

    // Iterate MAX_RAY_DEPTH times
    std::cout << "Trace" << std::endl;
    for (unsigned depth = 0; depth < MAX_RAY_DEPTH; ++depth)
    {
        Render<<< grid, block >>>(mLayers, depth, mDirections, mBases,
                                  spheres, size, mWidth, mHeight);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // Reassemble image
    std::cout << "Assemble" << std::endl;
    for (unsigned depth = MAX_RAY_DEPTH - 1; depth > 0; --depth)
    {
        Reassemble<<< grid, block >>>(mLayers, depth, mWidth, mHeight);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // Copy results back
    std::cout << "Output" << std::endl;
    cudaMemcpy(
        image, mLayers, mWidth*mHeight*sizeof(float3), cudaMemcpyDeviceToHost
    );
}

