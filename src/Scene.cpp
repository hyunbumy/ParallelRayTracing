#include <string>
#include <vector>
#include <iostream>
#include <random>
#include <time.h>

#include "Scene.h"    
#include "Sphere.h"
#include "Camera.h"
#include "Resources.h"
#include "Triangle.h"
#include "TriMesh.h"

using namespace std;

Scene::Scene(std::string filename, int select) {
    if(select == 1) { //basic 5 spheres
        Objects.push_back(new Sphere(Vector3( 0.0, -10004, -20), 10000, Vector3(0.20, 0.20, 0.20), 0, 0.0));
        Objects.push_back(new Sphere(Vector3( 0.0,      0, -20),     4, Vector3(1.00, 0.32, 0.36), 1, 0.5));
        Objects.push_back(new Sphere(Vector3( 5.0,     -1, -15),     2, Vector3(0.90, 0.76, 0.46), 1, 0.0));
        Objects.push_back(new Sphere(Vector3( 5.0,      0, -25),     3, Vector3(0.65, 0.77, 0.97), 1, 0.0));
        Objects.push_back(new Sphere(Vector3(-5.5,      0, -15),     3, Vector3(0.90, 0.90, 0.90), 1, 0.0));

        Objects.push_back(new Sphere(Vector3( 0.0,     20, -30),     3, Vector3(0.00, 0.00, 0.00), 0, 0.0, Vector3(3,3,3)));

        cam = Camera(Vector3(0, 0, 0), Vector3(0, 0, 0), Vector3(0, 0, 0), 1, 30, 3480, 2160);
    } else if(select == 2) { // 5 cows
        //Objects.push_back(new Sphere(Vector3( 0.0, -10004, -20), 10000, Vector3(0.20, 0.20, 0.20), 0, 0));

        Objects.push_back(new TriMesh("../mesh_files/cow.geo", Vector3(1.00, 0.32, 0.36), 1, 0));
        Objects.push_back(new TriMesh("../mesh_files/cow2.geo", Vector3(1.00, 0.32, 0.36), 1, 0));
        Objects.push_back(new TriMesh("../mesh_files/cow3.geo", Vector3(1.00, 0.32, 0.36), 1, 0));
        Objects.push_back(new TriMesh("../mesh_files/cow4.geo", Vector3(1.00, 0.32, 0.36), 1, 0));
        Objects.push_back(new TriMesh("../mesh_files/cow5.geo", Vector3(1.00, 0.32, 0.36), 1, 0));

        Objects.push_back(new Sphere(Vector3( 0.0, 0, 0), 2, Vector3(0.00, 0.00, 0.00), 0, 0.0, Vector3(3,3,3)));

        cam = Camera(Vector3(0, 6, 30), Vector3(0, 0, 0), Vector3(0.5, 0, 0), 1, 30, 3480, 2160);

    } else if(select == 3) { // banana
        //Objects.push_back(new Sphere(Vector3( 0.0, -10004, -20), 10000, Vector3(0.20, 0.20, 0.20), 0, 0.0));
        Objects.push_back(new TriMesh("../mesh_files/banana.dae", Vector3(1, 1, 0.25), 1, 0));
        Objects.push_back(new Sphere(Vector3( 0,     0.5, 0.3),     3 , Vector3(0.00, 0.00, 0.00), 0, 0.0, Vector3(7,7,7)));
                                //0.8
        cam = Camera(Vector3(-0.01, 0.20, 0.6), Vector3(0, 0, 0), Vector3(0, 0, 0), 1, 30, 3480, 2160);
    } else if(select == 4) { // 100 spheres
        int seed = 1362; //time(NULL) % 10000; //2057; //1362;
        cout << "seed: " << seed << endl;
        srand(seed);
        for(int i = 0; i < 100; i++) {
            float x = ( ((float) rand()) / RAND_MAX ) * 20 - 10; //ranges from -10 to 10
            float y = ( ((float) rand()) / RAND_MAX ) * 20 - 10;      //ranges from -2.5 to 2.5
            float z = ( ((float) rand()) / RAND_MAX ) * 100 * (-1) - 20; //ranges from -40 to -10

            float radius = ( ((float) rand()) / RAND_MAX ) * 2.5; //ranges from 0 to 3

            float r = ( ((float) rand()) / RAND_MAX ) * 0.6 + 0.4;
            float g = ( ((float) rand()) / RAND_MAX ) * 0.6 + 0.4;
            float b = ( ((float) rand()) / RAND_MAX ) * 0.6 + 0.4;

            Objects.push_back(new Sphere(Vector3(x, y, z), radius, Vector3(r, g, b), 1, 0.5));
        }
        Objects.push_back(new Sphere(Vector3( 0.0,     20, -30),     3, Vector3(0.00, 0.00, 0.00), 0, 0.0, Vector3(3,3,3)));

        cam = Camera(Vector3(0, 0, 5), Vector3(0, 0, 0), Vector3(0, 0, -0.1), 1, 30, 3480, 2160);
    }
}

Scene::~Scene() {
    for(auto obj : Objects) delete obj;
}