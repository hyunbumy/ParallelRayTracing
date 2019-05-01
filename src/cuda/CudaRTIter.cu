#include "CudaRTIter.h"
#include "CudaSphere.h"

#define MAX_TREE_SIZE 31

__device__
unsigned GetCurrentIndex(unsigned width, unsigned height)
{
    unsigned x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y*blockDim.y + threadIdx.y;
    return x + width * y;
}

__device__
float3 Trace(unsigned currInd, unsigned krInd, unsigned ktInd, 
             float3* directions, float3* bases, 
             float3* coefficients, unsigned depth,
             CudaObject** objects, int objSize)
{
    auto direction = directions[currInd];
    auto base = bases[currInd];
    auto coefficient = coefficients[currInd];

    // Default to no secondary rays
    if (depth < MAX_RAY_DEPTH-1)
    {
        coefficients[krInd] = make_float3(0,0,0);
        coefficients[ktInd] = make_float3(0,0,0);
    }

    // Ray Tracing finished
    if (coefficient.x == 0 && coefficient.y == 0 && coefficient.z == 0)
    {
        return make_float3(0,0,0);
    }

    // Get Intersection
    float tnear = INFINITY;
    const CudaObject* object = GetClosestObject(
        base, direction, tnear, objects, objSize
    );
    if (object == nullptr)
    {
        return make_float3(2,2,2)*coefficient;
    }

    float3 phit = base + direction * tnear;
    float3 nhit = object->CalculateHit(phit);
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

        // Update reflection ray
        bases[krInd] = intersection;
        directions[krInd] = normalize(reflDir);
        coefficients[krInd] = object->surfaceColor * fresnelEffect * coefficient;

        // Check refraction
        if (object->transparency)
        {
            float ior = 1.1, eta = (inside) ? ior : 1 / ior; // are we inside or outside the surface?
            float cosi = -1 * dot(nhit, direction);
            float k = 1 - eta * eta * (1 - cosi * cosi);
            float3 refrDir = direction * eta + nhit * (eta *  cosi - sqrtf(k));
            
            // Update refraction ray
            intersection = phit - nhit * bias;
            bases[ktInd] = intersection;
            directions[ktInd] = normalize(refrDir);
            coefficients[ktInd] = object->surfaceColor * (1-fresnelEffect) *
                                  coefficient * object->transparency;
        }

        return make_float3(0,0,0);
    }

    // Diffuse object, compute illumination
    for (unsigned i = 0; i < objSize; ++i)
    {
        if (objects[i]->emissionColor.x > 0)
        {
            // this is a light
            float3 transmission = make_float3(1,1,1);
            float3 lightDirection = -1 * objects[i]->CalculateHit(phit);
            lightDirection = normalize(lightDirection);

            // Check for obstruction
            for (unsigned j = 0; j < objSize; ++j) {
                if (i != j) {
                    float t0, t1;
                    if (objects[j]->Intersect(intersection, lightDirection, t0, t1))
                    {
                        transmission = make_float3(0,0,0);
                        break;
                    }
                }
            }
            surfaceColor += object->surfaceColor * transmission *
                            fmaxf(float(0), dot(nhit, lightDirection)) *
                            objects[i]->emissionColor;
        }
    }

    // If diffuse object, stop ray tracing
    return (surfaceColor + object->emissionColor)*coefficient;
}

__global__
void Initialize(float3* coefficients, float3* directions, float3* bases, 
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

    // Coeficients
    coefficients[i] = make_float3(1,1,1);
}

__global__
void Render(float3* layers, unsigned depth,
            float3* coefficients, float3* directions, float3* bases,
            CudaObject** objects, int objSize,
            unsigned width, unsigned height)
{
    unsigned i = GetCurrentIndex(width, height);

    unsigned start = powf(2, depth);
    unsigned end = start * 2 - 1;
    --start;
    for (int index = start; index < end; ++index)
    {
        unsigned currInd = i + width*height*index;
        unsigned krInd = i + width*height*(2*index+1);
        unsigned ktInd = i + width*height*(2*index+2);
        layers[currInd] = Trace(
            currInd, krInd, ktInd, directions, bases, coefficients, depth, objects, objSize
        );
    }
}

__global__
void Reassemble(float3* layers, unsigned depth, 
                unsigned width, unsigned height)
{
    unsigned i = GetCurrentIndex(width, height);

    unsigned start = powf(2, depth);
    unsigned end = start * 2 - 1;
    --start;
    for (int index = start; index < end; ++index)
    {
        layers[i] += layers[i+width*height*index];
    }
}

__global__
void AllocateIter(CudaObject** objects)
{
    new (objects[0]) CudaSphere(make_float3(0, -10004, -20), 10000, make_float3(0.2, 0.2, 0.2), 0, 0);
    new (objects[1]) CudaSphere(make_float3(0, 0, -20), 4.0, make_float3(1.0, 0.32, 0.36), 1, 0.5);
    new (objects[2]) CudaSphere(make_float3(5, -1, -15), 2, make_float3(0.9, 0.76, 0.46), 1, 0.0);
    new (objects[3]) CudaSphere(make_float3(5, 0, -25), 3, make_float3(0.65, 0.77, 0.97), 1, 0.0);
    new (objects[4]) CudaSphere(make_float3(-5.5, 0, -15), 3, make_float3(0.9, 0.9, 0.9), 1, 0.0);
    new (objects[5]) CudaSphere(make_float3(0.0, 20, -30), 3, make_float3(0, 0, 0), 0, 0.0f, make_float3(3,3,3));
}

CudaRTIter::CudaRTIter(unsigned width, unsigned height)
    :mWidth(width)
    ,mHeight(height)
{
    checkCudaErrors(
        cudaMalloc(&mLayers, MAX_TREE_SIZE*width*height*sizeof(float3))
    );

    checkCudaErrors(
        cudaMalloc(&mDirections, MAX_TREE_SIZE*width*height*sizeof(float3))
    );

    checkCudaErrors(
        cudaMalloc(&mBases, MAX_TREE_SIZE*width*height*sizeof(float3))
    );
    checkCudaErrors(
        cudaMalloc(&mCoefficients, MAX_TREE_SIZE*width*height*sizeof(float3))
    );
}

CudaRTIter::~CudaRTIter()
{
    // free CUDA memory
    checkCudaErrors(cudaFree(mLayers));
    checkCudaErrors(cudaFree(mDirections));
    checkCudaErrors(cudaFree(mBases));
    checkCudaErrors(cudaFree(mCoefficients));
}

void CudaRTIter::RenderWrapper(float3* image)
{
    CudaObject** objects;
    int size = 6;
    checkCudaErrors(cudaMallocManaged(&objects, size*sizeof(CudaObject*)));
    checkCudaErrors(cudaMallocManaged((void**)&objects, size*sizeof(CudaObject*)));
    for (int i = 0; i < size; ++i)
    {
        checkCudaErrors(cudaMallocManaged((void**)&objects[i], sizeof(CudaSphere)));
    }
    AllocateIter<<<1,1>>>(objects);

    // dim3 is CUDA specific type, block and grid are required to schedule CUDA threads over streaming multiprocessors
    dim3 block(8, 8, 1);   
    dim3 grid(mWidth / block.x, mHeight / block.y, 1);

    std::cout << "Initialize" << std::endl;
    // schedule threads on device and launch CUDA kernel from host
    Initialize<<< grid, block >>>(mCoefficients, mDirections, mBases, mWidth, mHeight);
    checkCudaErrors(cudaDeviceSynchronize());

    // Iterate MAX_RAY_DEPTH times
    std::cout << "Trace" << std::endl;
    for (unsigned depth = 0; depth < MAX_RAY_DEPTH; ++depth)
    {
        Render<<< grid, block >>>(mLayers, depth, 
                                  mCoefficients, mDirections, mBases,
                                  objects, size, mWidth, mHeight);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    // Reassemble image
    std::cout << "Assemble" << std::endl;
    for (unsigned depth = MAX_RAY_DEPTH - 1; depth > 0; --depth)
    {
        Reassemble<<< grid, block >>>(mLayers, depth, mWidth, mHeight);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy results back
    std::cout << "Output" << std::endl;
    cudaMemcpy(
        image, mLayers, mWidth*mHeight*sizeof(float3), cudaMemcpyDeviceToHost
    );

    // Free up memory
    for (int i = 0; i < size; ++i)
    {
        checkCudaErrors(cudaFree(objects[i]));
    }
    checkCudaErrors(cudaFree(objects));
}

