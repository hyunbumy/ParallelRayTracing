// [header]
// A very basic raytracer example.
// [/header]
// [compile]
// c++ -o raytracer -O3 -Wall raytracer.cpp
// [/compile]
// [ignore]
// Copyright (C) 2012  www.scratchapixel.com
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// [/ignore]
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <iostream>
#include <cassert>
#include <mpi.h>
#include <vector>

#include "RayTracer.h"
#include "Output.h"

using namespace std;

//[comment]
// This variable controls the maximum recursion depth
//[/comment]
#define MAX_RAY_DEPTH 5

//[comment]
// This is the main trace function. It takes a ray as argument (defined by its origin
// and direction). We test if this ray intersects any of the geometry in the scene.
// If the ray intersects an object, we compute the intersection point, the normal
// at the intersection point, and shade this point using this information.
// Shading depends on the surface property (is it transparent, reflective, diffuse).
// The function returns a color for the ray. If the ray intersects an object that
// is the color of the object at the intersection point, otherwise it returns
// the background color.
//[/comment]
Vector3 RayTracer::Trace(
    const Vector3 &rayorig,
    const Vector3 &raydir,
    const std::vector<Object*> &objects,
    const int &depth)
{
    //if (raydir.length() != 1) std::cerr << "Error " << raydir << std::endl;
    float tnear = INFINITY;
    const Object* object = NULL;
    // find intersection of this ray with the sphere in the scene
    for (unsigned i = 0; i < objects.size(); ++i) {
        float t0 = INFINITY, t1 = INFINITY;
        if (objects[i]->intersect(rayorig, raydir, t0, t1)) {
            if (t0 < 0) t0 = t1;
            if (t0 < tnear) {
                tnear = t0;
                object = objects[i];
            }
        }
    }
    // if there's no intersection return black or background color
    if (!object) return Vector3(2, 2, 2);
    Vector3 surfaceColor = Vector3::Zero; // color of the ray/surfaceof the object intersected by the ray
    Vector3 phit = rayorig + raydir * tnear; // point of intersection
    Vector3 nhit = object->calculateHit(phit); // normal at the intersection point
    nhit.Normalize(); // normalize normal direction
    // If the normal and the view direction are not opposite to each other
    // reverse the normal direction. That also means we are inside the sphere so set
    // the inside bool to true. Finally reverse the sign of IdotN which we want
    // positive.
    float bias = 1e-4; // add some bias to the point from which we will be tracing
    bool inside = false;
    if (Vector3::Dot(raydir, nhit) > 0)
    {
        nhit *= -1;
        inside = true;
    }
    if ((object->transparency > 0 || object->reflection > 0) && depth < MAX_RAY_DEPTH) {
        float facingratio = -1 * Vector3::Dot(raydir, nhit);
        // change the mix value to tweak the effect
        float fresneleffect = Math::Mix(pow(1 - facingratio, 3), 1, 0.1);
        // compute reflection direction (not need to normalize because all vectors
        // are already normalized)
        Vector3 refldir = raydir - nhit * 2 * Vector3::Dot(raydir, nhit);
        refldir.Normalize();
        Vector3 reflection = Trace(phit + nhit * bias, refldir, objects, depth + 1);
        Vector3 refraction = Vector3::Zero;
        // if the sphere is also transparent compute refraction ray (transmission)
        if (object->transparency) {
            float ior = 1.1, eta = (inside) ? ior : 1 / ior; // are we inside or outside the surface?
            float cosi = -1 * Vector3::Dot(nhit, raydir);
            float k = 1 - eta * eta * (1 - cosi * cosi);
            Vector3 refrdir = raydir * eta + nhit * (eta *  cosi - sqrt(k));
            refrdir.Normalize();
            refraction = Trace(phit - nhit * bias, refrdir, objects, depth + 1);
        }
        // the result is a mix of reflection and refraction (if the sphere is transparent)
        surfaceColor = 
            (reflection * fresneleffect * object->surfaceColor) +
            (refraction * (1 - fresneleffect) * object->transparency * object->surfaceColor);
        return surfaceColor;
    }
    else {
        // it's a diffuse object, no need to raytrace any further
        for (unsigned i = 0; i < objects.size(); ++i) {
            if (objects[i]->emissionColor.x > 0) {
                // this is a light
                Vector3 transmission = Vector3(1,1,1);
                Vector3 lightDirection = -1 * objects[i]->calculateHit(phit);
                lightDirection.Normalize();
                for (unsigned j = 0; j < objects.size(); ++j) {
                    if (i != j) {
                        float t0, t1;
                        if (objects[j]->intersect(phit + nhit * bias, lightDirection, t0, t1)) {
                            transmission = Vector3::Zero;
                            break;
                        }
                    }
                }
                surfaceColor += object->surfaceColor * transmission *
                std::max(float(0), Vector3::Dot(nhit, lightDirection)) * objects[i]->emissionColor;
            }
        }
        return surfaceColor + object->emissionColor;
    }
}

MPI_Datatype getDataTransferType() {
    const int nitems=3;
    int          blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
    MPI_Datatype mpi_vector3;
    MPI_Aint     offsets[3];

    offsets[0] = offsetof(Vector3, x);
    offsets[1] = offsetof(Vector3, y);
    offsets[2] = offsetof(Vector3, z);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_vector3);
    MPI_Type_commit(&mpi_vector3);
    return mpi_vector3;
}

void RayTracer::master(int &size, int &rank, Scene &scene) {
    //printf("master\n");
    MPI_Status status;
    MPI_Datatype mpi_vector3 = getDataTransferType();
    // create holding vector
    // distribute initial work to workers
    // loop and retrieve until work done
    //      receive row number
    //      receive data
    //      send more work
    // send termination to workers

    unsigned width = scene.cam.width, height = scene.cam.height;
    std::vector<std::vector<Vector3> > image = std::vector<std::vector<Vector3> >(
        height, std::vector<Vector3>(width)
    );

    int curr_row = 0;

    //distributed initial work
    for(int i = 1; i < size; i++) {
        MPI_Send(&curr_row, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        curr_row++;
    }
    int recv_row;
    int done = -1;
    while(curr_row < height) {
        //MPI_REC
        MPI_Recv(&recv_row, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        //printf("master received row %d\n", recv_row);

        //MPI_REC
        MPI_Recv(&image[recv_row][0], width, mpi_vector3, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        //printf("master received row data from %d\n", recv_row);

        //send more work
        MPI_Send(&curr_row, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
        
        curr_row++;
    }

    //receive and terminate worker
    for(int i = 1; i < size; i++) {
        MPI_Recv(&recv_row, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&image[recv_row][0], width, mpi_vector3, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Send(&done, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
    }

    //printf("finished receiving all data\n");
    Output::OutputPPM(image);
    return;
}



void RayTracer::worker(int &rank, Scene &scene) {
    MPI_Status status;
    MPI_Datatype mpi_vector3 = getDataTransferType();

    unsigned width = scene.cam.width, height = scene.cam.height;

    float invWidth = 1 / float(width), invHeight = 1 / float(height);
    float fov = scene.cam.fov, aspectratio = width / float(height);
    float angle = tan(M_PI * 0.5 * fov / 180.);

    vector<Vector3> row_data = vector<Vector3>(scene.cam.width);

    int row = 0;
    while(true) {
        MPI_Recv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        //printf("Rank: %d, computing row: %d\n", rank, row);
        if(row == -1) break;
        for(int i = 0; i < scene.cam.width; i++) {

            //create primary ray
            float xx = (2 * ((i + 0.5) * invWidth) - 1) * angle * aspectratio;
            float yy = (1 - 2 * ((row + 0.5) * invHeight)) * angle;
            float zz = -1;
            Vector3 raydir(xx, yy, zz);
            raydir = raydir - scene.cam.direction;
            raydir.Normalize();

            //trace primary ray
            row_data[i] = Trace(scene.cam.position, raydir, scene.Objects, 0);
        }
        //printf("sending row num\n");
        MPI_Send(&row, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        //printf("sending row data\n");
        MPI_Send(&row_data[0], width, mpi_vector3, 0, 1, MPI_COMM_WORLD);
    }
    //printf("worker exiting\n");
    return;
}