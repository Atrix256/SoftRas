#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <vector>

#define THIRTEEN_IMPLEMENTATION
#include "external/thirteen.h"

#define STB_IMAGE_IMPLEMENTATION
#include "external/stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb/stb_image_write.h"

#include "MathUtils.h"

#define MULTI_THREADED() 0 // TODO: turn to true when working

static const float c_epsilon = 3.0f;

// In UV space
Point2D g_mesh[] =
{
//    {0.0f, 0.0f},
//    {1.0f, 0.0f},
//    {1.0f, 1.0f},

    {0.3f, 0.3f},
    {0.6f, 0.3f},
    {0.6f, 0.6f},

//    {0.6f, 0.6f},
//    {0.6f, 0.3f},
//    {0.3f, 0.3f},
};

float SoftCoverage(const Point2D& P, const Point2D& A, const Point2D& B, const Point2D& C)
{
    float e0 = Signed2DTriArea(A, B, P);
    float e1 = Signed2DTriArea(B, C, P);
    float e2 = Signed2DTriArea(C, A, P);

    return Sigmoid(e0 / c_epsilon) * Sigmoid(e1 / c_epsilon) * Sigmoid(e2 / c_epsilon);
}

void RasterizeMesh(unsigned char* pixels, unsigned int width, unsigned int height)
{
    // Clear to black
    memset(pixels, 0, width * height * 4);

    // transform the mesh into pixel coordinates
    static std::vector<Point2D> screenPoints;
    screenPoints.resize(_countof(g_mesh));
    #if MULTI_THREADED()
    #pragma omp parallel for
    #endif
    for (int index = 0; index < _countof(g_mesh); ++index)
    {
        screenPoints[index].x = g_mesh[index].x * float(width);
        screenPoints[index].y = g_mesh[index].y * float(height);
    }

    // rasterize
    #if MULTI_THREADED()
    #pragma omp parallel for
    #endif
    for (int iy = 0; iy < (int)height; ++iy)
    {
        for (int ix = 0; ix < (int)width; ++ix)
        {
            for (int triangleIndex = 0; triangleIndex < _countof(g_mesh) / 3; ++triangleIndex)
            {

                if (ix == (width/2) && iy == (height / 2))
                {
                    int ijkl = 0;
                }

                float coverage = SoftCoverage(Point2D{ float(ix), float(iy) }, screenPoints[triangleIndex * 3 + 0], screenPoints[triangleIndex * 3 + 1], screenPoints[triangleIndex * 3 + 2]);

                pixels[((iy * width) + ix) * 4 + 0] = (unsigned char)Clamp<int>((int)(coverage * 255.0f), 0, 255);
                pixels[((iy * width) + ix) * 4 + 1] = 0;
                pixels[((iy * width) + ix) * 4 + 2] = 0;
                pixels[((iy * width) + ix) * 4 + 3] = 255;
            }
        }
    }
}

int main(int argc, char** argv)
{
    unsigned char* pixels = Thirteen::Init(800, 600);
    if (!pixels)
        return 1;

    // Go until window is closed or escape is pressed
    while (Thirteen::Render() && !Thirteen::GetKey(VK_ESCAPE))
    {
        RasterizeMesh(pixels, Thirteen::GetWidth(), Thirteen::GetHeight());
    }

    Thirteen::Shutdown();
    return 0;
}
/*
TODO:
- 2d first then 3d
- make optimization work for... vertex positions, material param (color?), lights, camera
- soft ras.
- link to paper and the blog post that talks about more advanced stuff

Notes:
The core idea is that rasterization has a binary hard edge over space, and occlusion via the depth buffer is a binary hard edge over depth.
By making both of these soft, it allows differentiation.

Paper:
https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Soft_Rasterizer_A_Differentiable_Renderer_for_Image-Based_3D_Reasoning_ICCV_2019_paper.pdf

Link to this for slightly newer methods of differentiable rasterization:
https://jjbannister.github.io/tinydiffrast/

*/
