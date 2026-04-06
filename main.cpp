#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <vector>
#include <algorithm>

#define THIRTEEN_IMPLEMENTATION
#include "external/thirteen.h"

#define STB_IMAGE_IMPLEMENTATION
#include "external/stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb/stb_image_write.h"

#include "MathUtils.h"

#define MULTI_THREADED() 0 // TODO: turn to true when working

static const float c_sigma = 3.0f; // TODO: paper uses 1x10^-4. what should we use?

// Constants from equation 3
static const float c_epsilon = 1e-4f; // TODO: this is not specified in the paper
static const float c_gamma = 1e-4f;

static const Point3D c_backgroundColor = { 0.2f, 0.2f, 0.2f };

struct GBuffer
{
    float coverage = 0.0f;
    float depth = 0.0f;
    Point3D color = { 0.0f, 0.0f, 0.0f };
};

struct Vertex
{
    Point3D pos;
    Point3D color;
};

// In UV space
Vertex g_mesh[] =
{
    { {0.3f, 0.3f, 0.001f}, {1.0f, 0.0f, 0.0f} },
    { {0.6f, 0.3f, 0.001f}, {0.0f, 1.0f, 0.0f} },
    { {0.6f, 0.6f, 0.001f}, {0.0f, 0.0f, 1.0f} },

    { {0.1f, 0.1f, 0.002f}, {1.0f, 0.0f, 0.0f} },
    { {0.2f, 0.1f, 0.002f}, {0.0f, 1.0f, 0.0f} },
    { {0.2f, 0.2f, 0.002f}, {0.0f, 0.0f, 1.0f} },
};

float CrossProduct2D(float x1, float y1, float x2, float y2)
{
    return x1 * y2 - x2 * y1;
}

// TODO: i think we can calculate this in sdTriangle
Point3D CalculateBarycentricCoordinates(Point2D A, Point2D B, Point2D C, Point2D P)
{
    Point3D ret;
    float& u = ret.x;
    float& v = ret.y;
    float& w = ret.z;

    // Vectors for the full triangle
    float ax = B.x - A.x;
    float ay = B.y - A.y;
    float bx = C.x - A.x;
    float by = C.y - A.y;

    // Vectors for the point P relative to A
    float px = P.x - A.x;
    float py = P.y - A.y;

    float totalArea = CrossProduct2D(ax, ay, bx, by);

    // Check for a degenerate triangle
    if (std::abs(totalArea) < 1e-9) {
        u = v = w = 0.0f;
        return ret;
    }

    // Calculate the coordinates
    v = CrossProduct2D(px, py, bx, by) / totalArea; // Weight for B
    w = CrossProduct2D(ax, ay, px, py) / totalArea; // Weight for C
    u = 1.0f - v - w;                               // Weight for A

    return ret;
}

// Triangle SDF from https://iquilezles.org/articles/distfunctions2d/
// Returns positive values if outside the triangle, negative values if inside the triangle
float sdTriangle(const Point2D& p, const Point2D& p0, const Point2D& p1, const Point2D& p2)
{
    Point2D e0 = p1 - p0, e1 = p2 - p1, e2 = p0 - p2;
    Point2D v0 = p - p0, v1 = p - p1, v2 = p - p2;
    Point2D pq0 = v0 - e0 * Clamp(Dot(v0, e0) / Dot(e0, e0), 0.0f, 1.0f);
    Point2D pq1 = v1 - e1 * Clamp(Dot(v1, e1) / Dot(e1, e1), 0.0f, 1.0f);
    Point2D pq2 = v2 - e2 * Clamp(Dot(v2, e2) / Dot(e2, e2), 0.0f, 1.0f);
    float s = Sign(e0.x * e2.y - e0.y * e2.x);
    Point2D d =
        std::min(
            std::min(
                Point2D{ Dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x) },
                Point2D{ Dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x) }
            ),
            Point2D{ Dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x) }
        )
    ;
    return -sqrt(d.x) * Sign(d.y);
}

float SoftCoverage(const Point2D& P, const Point2D& A, const Point2D& B, const Point2D& C)
{
    // Equation 1
    float sdf = -sdTriangle(P, A, B, C);
    return Sigmoid(Sign(sdf) * sdf * sdf / c_sigma);
}

void RasterizeMesh(unsigned char* pixels, std::vector<GBuffer>* gBuffer, unsigned int width, unsigned int height)
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
        screenPoints[index].x = g_mesh[index].pos.x * float(width);
        screenPoints[index].y = g_mesh[index].pos.y * float(height);
    }

    // rasterize
    #if MULTI_THREADED()
    #pragma omp parallel for
    #endif
    for (int iy = 0; iy < (int)height; ++iy)
    {
        for (int ix = 0; ix < (int)width; ++ix)
        {
            int pixelIndex = iy * width + ix;

            std::vector<GBuffer>& gb = gBuffer[pixelIndex];
            gb.resize(0);

            // TODO: temp. Or maybe leave this! it makes a breakpoint on the pixel that you click
            if (Thirteen::GetMouseButton(0) && !Thirteen::GetMouseButtonLastFrame(0))
            {
                int mouseX, mouseY;
                Thirteen::GetMousePosition(mouseX, mouseY);
                if (ix == mouseX && iy == mouseY)
                {
                    int ijkl = 0;
                }
            }

            // TODO: may not need to store all gbuffer entries, but just calculate gradients right here per pixel and store those, with the final color.

            for (int triangleIndex = 0; triangleIndex < _countof(g_mesh) / 3; ++triangleIndex)
            {
                GBuffer newGB;

                const Vertex& vA = g_mesh[triangleIndex * 3 + 0];
                const Vertex& vB = g_mesh[triangleIndex * 3 + 1];
                const Vertex& vC = g_mesh[triangleIndex * 3 + 2];

                newGB.coverage = SoftCoverage({ float(ix), float(iy) }, screenPoints[triangleIndex * 3 + 0], screenPoints[triangleIndex * 3 + 1], screenPoints[triangleIndex * 3 + 2]);

                Point3D uvw = CalculateBarycentricCoordinates(screenPoints[triangleIndex * 3 + 0], screenPoints[triangleIndex * 3 + 1], screenPoints[triangleIndex * 3 + 2], Point2D{ float(ix), float(iy) });

                // The paper says they clamp uvw between 0 and 1 and then they renormalize it to sum to 1
                uvw = Clamp(uvw, 0.0f, 1.0f);
				float sum = uvw.x + uvw.y + uvw.z;
				uvw.x /= sum;
				uvw.y /= sum;
				uvw.z /= sum;

                newGB.color = (vA.color * uvw.x + vB.color * uvw.y + vC.color * uvw.z);
                newGB.depth = (vA.pos.z * uvw.x + vB.pos.z * uvw.y + vC.pos.z * uvw.z);

                gb.push_back(newGB);
            }

			Point3D pixelColor = { 0.0f, 0.0f, 0.0f };

            // Calculate the denominator in equation 3
            float weightDenom = std::exp(c_epsilon / c_gamma);
            for (const GBuffer& entry : gb)
                weightDenom += entry.coverage * std::exp(entry.depth / c_gamma);

            // Calculate equation 2
			float totalWeight = 0.0f;
            for (const GBuffer& entry : gb)
            {
				float weight = entry.coverage * std::exp(entry.depth / c_gamma) / weightDenom;
				pixelColor = pixelColor + entry.color * weight;
				totalWeight += weight;
            }

			float backgroundWeight = 1.0f - totalWeight;
			pixelColor = pixelColor + c_backgroundColor * backgroundWeight;

            pixels[pixelIndex * 4 + 0] = (unsigned char)Clamp(pixelColor.x * 255.0f, 0.0f, 255.0f);
            pixels[pixelIndex * 4 + 1] = (unsigned char)Clamp(pixelColor.y * 255.0f, 0.0f, 255.0f);
            pixels[pixelIndex * 4 + 2] = (unsigned char)Clamp(pixelColor.z * 255.0f, 0.0f, 255.0f);
            pixels[pixelIndex * 4 + 3] = 255;
        }
    }
}

int main(int argc, char** argv)
{
    unsigned char* pixels = Thirteen::Init(800, 600);
    if (!pixels)
        return 1;

    std::vector<std::vector<GBuffer>> gBuffer(Thirteen::GetWidth() * Thirteen::GetHeight());

    while (Thirteen::Render() && !Thirteen::GetKey(VK_ESCAPE))
    {
        RasterizeMesh(pixels, gBuffer.data(), Thirteen::GetWidth(), Thirteen::GetHeight());

        if (Thirteen::GetKey('V') && !Thirteen::GetKeyLastFrame('V'))
            Thirteen::SetVSync(!Thirteen::GetVSync());
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
