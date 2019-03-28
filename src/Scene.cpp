#include <string>
#include <vector>
#include <iostream>

#include "Scene.h"    
#include "Sphere.h"
#include "Camera.h"
#include "Math.h"

using namespace std;

Scene::Scene(std::string filename) {
    Objects.push_back(new Sphere(Vector3( 0.0, -10004, -20), 10000, Vector3(0.20, 0.20, 0.20), 0, 0.0));
    Objects.push_back(new Sphere(Vector3( 0.0,      0, -20),     4, Vector3(1.00, 0.32, 0.36), 1, 0.5));
    Objects.push_back(new Sphere(Vector3( 5.0,     -1, -15),     2, Vector3(0.90, 0.76, 0.46), 1, 0.0));
    Objects.push_back(new Sphere(Vector3( 5.0,      0, -25),     3, Vector3(0.65, 0.77, 0.97), 1, 0.0));
    Objects.push_back(new Sphere(Vector3(-5.5,      0, -15),     3, Vector3(0.90, 0.90, 0.90), 1, 0.0));
    // light
    Objects.push_back(new Sphere(Vector3( 0.0,     20, -30),     3, Vector3(0.00, 0.00, 0.00), 0, 0.0, Vector3(3,3,3)));

    //cam = Camera(Vector3(10, 30, 15), Vector3(0.35, 1, 0), 50, 640, 480);
    cam = Camera(Vector3(0, 20, 30), Vector3(0, 0.3, 0), 50, 640, 480);
}

Scene::~Scene() {
    for(auto obj : Objects) delete obj;
}