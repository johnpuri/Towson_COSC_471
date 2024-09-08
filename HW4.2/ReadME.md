## Programming Assignment 3: Understanding of Output
## Overview
In this programming assignment, the goal is to simulate modern graphics technology by adding an Object Loader, Vertex Loader, Fragment Loader, and support for texture mapping. The code skeleton already contains some functionalities, but modifications are required to implement Blinn-Phong reflection model, Texture Shading Fragment Shader, Bump mapping, and Displacement mapping.

## Tasks
To complete this programming assignment, the following tasks need to be performed:

1. Modify the **rasterize_triangle**() function in rasterizer.cpp to interpolate normal, color, and texture values similar to the previous assignment.
2. Modify the **get_projection_matrix**() function in main.cpp to implement the previous projection matrix and check the normal results.
3. Modify the **phong_fragment_shader**() function in main.cpp to implement the Blinn-Phong reflection model to calculate Fragment Color.
4. Modify the **texture_fragment_shader**() function in main.cpp to implement Texture Shading Fragment Shader based on the implementation of the Blinn-Phong model.
5. Modify the **bump_fragment_shader**() function in main.cpp to implement Bump mapping based on the implementation of the Blinn-Phong model.
6. Modify the **displacement_fragment_shader**() function in main.cpp to implement Displacement mapping based on the Bump mapping.

## Compiling
To compile and run the code, follow these steps:

1. Download the skeleton code provided for this assignment.
2. Navigate to the root directory of the downloaded code and create a new directory named build.
Change to the build directory and run the command cmake ...
3. Run make to generate an executable file named Rasterizer.
4. To run the code, provide the image name as the first parameter and the type of shader as the second parameter. The following types of shaders are supported:
* texture: runs the code with texture shader.
* normal: runs the code with normal shader.
* phong: runs the code with Blinn-Phong shader.
* bump: runs the code with bump shader.
* displacement: runs the code with displacement shader.
For example, to run the code with the Blinn-Phong shader and save the output as output.png, use the following command:
** ./Rasterizer output.png phong **

## Skeleton codes
Compared to the previous assignment, some modifications have been made to the skeleton codes:

1. A third-party .obj file loader has been introduced to read in complicated models, and this library file is in the OBJ_Loader.h file. The loader passes a Vector named TriangleList, in which each triangle has corresponding normal and texture coordinates. Besides, the texture related to the model will also be loaded. Note that if you try to load in another model, you have to change the model directory and path manually.

2. A new Texture class has been introduced to generate textures from images, and an interface to check texture colors: Vector3f getColor(float u, float v).

3. A header file Shader.hpp has been built and defined fragment_shader_payload, which includes parameters for Fragment Shader. Now main.cpp has three Fragment Shaders, which are fragment_shader (shader using normal) and two shaders that need to be implemented.

4. The main rendering pipeline starts with rasterizer::draw(std::vector<Triangle> &TriangleList). Then some transformations are made, which is
