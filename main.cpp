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

static const float c_sigma = 1e-4f;

// Constants from equation 3
static const float c_epsilon = 1e-4f; // TODO: this is not specified in the paper
static const float c_gamma = 1e-4f;

static const float c_nearPlane = 0.1f;
static const float c_farPlane = 20.0f;

static const float c_minimumCoverage = 1e-6f;

static const Point3D c_backgroundColor = { 0.2f, 0.2f, 0.2f };

struct Vertex
{
    Point3D pos;
    Point3D color;
};

// In UV space
Vertex g_mesh[] =
{
    { {0.3f, 0.3f, 10.0f}, {0.0f, 1.0f, 0.0f} },
    { {0.6f, 0.3f, 10.0f}, {0.0f, 1.0f, 0.0f} },
    { {0.6f, 0.6f, 10.0f}, {0.0f, 1.0f, 0.0f} },

    { {0.2f, 0.2f, 11.0f}, {1.0f, 0.0f, 0.0f} },
    { {0.5f, 0.2f, 11.0f}, {1.0f, 0.0f, 0.0f} },
    { {0.5f, 0.6f, 11.0f}, {1.0f, 0.0f, 0.0f} },
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

void RasterizeMesh(unsigned char* pixels, unsigned int width, unsigned int height)
{
    // Clear to black
    memset(pixels, 0, width * height * 4);

    // transform the mesh into pixel coordinates
    static std::vector<Point3D> screenPoints;
    screenPoints.resize(_countof(g_mesh));
    #if MULTI_THREADED()
    #pragma omp parallel for
    #endif
    for (int index = 0; index < _countof(g_mesh); ++index)
    {
        screenPoints[index].x = g_mesh[index].pos.x * float(width);
        screenPoints[index].y = g_mesh[index].pos.y * float(height);
        screenPoints[index].z = (c_farPlane - g_mesh[index].pos.z) / (c_farPlane - c_nearPlane);
    }

    // rasterize
    #if MULTI_THREADED()
    #pragma omp parallel for
    #endif
    for (int iy = 0; iy < (int)height; ++iy)
    {
        struct PxInfo
        {
            float coverage = 0.0f;
            float depth = 0.0f;
            Point3D color = { 0.0f, 0.0f, 0.0f };
        };
        std::vector<PxInfo> pxInfo;

        for (int ix = 0; ix < (int)width; ++ix)
        {
            pxInfo.clear();
            int pixelIndex = iy * width + ix;

            // If you want to debug a specific pixel, turn off MULTI_THREADED(), put a breakpoint in here, and click the pixel you want to debug.
            if (Thirteen::GetMouseButton(0) && !Thirteen::GetMouseButtonLastFrame(0))
            {
                int mouseX, mouseY;
                Thirteen::GetMousePosition(mouseX, mouseY);
                if (ix == mouseX && iy == mouseY)
                {
                    int ijkl = 0;
                }
            }

            for (int triangleIndex = 0; triangleIndex < _countof(g_mesh) / 3; ++triangleIndex)
            {
                PxInfo newPx;

                const Vertex& vA = g_mesh[triangleIndex * 3 + 0];
                const Vertex& vB = g_mesh[triangleIndex * 3 + 1];
                const Vertex& vC = g_mesh[triangleIndex * 3 + 2];

                const Point3D& sA = screenPoints[triangleIndex * 3 + 0];
                const Point3D& sB = screenPoints[triangleIndex * 3 + 1];
                const Point3D& sC = screenPoints[triangleIndex * 3 + 2];

                Point2D resolution = Point2D{ float(width), float(height) };

                Point2D sA2DNormed = (Point2D{ sA.x, sA.y } + 0.5f) / resolution;
                Point2D sB2DNormed = (Point2D{ sB.x, sB.y } + 0.5f) / resolution;
                Point2D sC2DNormed = (Point2D{ sC.x, sC.y } + 0.5f) / resolution;
				Point2D pixelNormed = (Point2D{ float(ix), float(iy) } + 0.5f) / resolution;

                newPx.coverage = SoftCoverage(pixelNormed, sA2DNormed, sB2DNormed, sC2DNormed);

                if (newPx.coverage < c_minimumCoverage)
                    continue;

                Point3D uvw = CalculateBarycentricCoordinates(sA2DNormed, sB2DNormed, sC2DNormed, pixelNormed);

                // The paper says they clamp uvw between 0 and 1 and then they renormalize it to sum to 1
                uvw = Clamp(uvw, 0.0f, 1.0f);
                float sum = uvw.x + uvw.y + uvw.z;
                uvw.x /= sum;
                uvw.y /= sum;
                uvw.z /= sum;

                newPx.color = (vA.color * uvw.x + vB.color * uvw.y + vC.color * uvw.z);
                newPx.depth = (sA.z * uvw.x + sB.z * uvw.y + sC.z * uvw.z);

                pxInfo.push_back(newPx);
            }

            Point3D pixelColor = { 0.0f, 0.0f, 0.0f };
            float totalWeight = 0.0f;

            // If this pixel is covered by any triangles
            if (pxInfo.size() > 0)
            {
                // Track the max exponent across all triangles
                float softmax_max = c_epsilon / c_gamma;  // from background term
                for (const PxInfo& entry : pxInfo)
                    softmax_max = std::max(softmax_max, entry.depth / c_gamma);

                // Compute denominator with max subtracted (numerically stable)
                float weightDenom = std::exp(c_epsilon / c_gamma - softmax_max);
                for (const PxInfo& entry : pxInfo)
                    weightDenom += entry.coverage * std::exp(entry.depth / c_gamma - softmax_max);

                // Weights — the softmax_max cancels in numerator/denominator
                for (const PxInfo& entry : pxInfo)
                {
                    float weight = entry.coverage * std::exp(entry.depth / c_gamma - softmax_max) / weightDenom;
                    pixelColor = pixelColor + entry.color * weight;
                    totalWeight += weight;
                }
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

    while (Thirteen::Render() && !Thirteen::GetKey(VK_ESCAPE))
    {
        RasterizeMesh(pixels, Thirteen::GetWidth(), Thirteen::GetHeight());

        if (Thirteen::GetKey('V') && !Thirteen::GetKeyLastFrame('V'))
            Thirteen::SetVSync(!Thirteen::GetVSync());
    }

    Thirteen::Shutdown();
    return 0;
}
/*
Gradients next?


TODO:
- 2d first then 3d
- make optimization work for... vertex positions, material param (color?), lights, camera
- soft ras.
- link to paper and the blog post that talks about more advanced stuff
? should you do the Silhouette aggregate function? "It's used with a separate silhouette loss for unsupervised reconstruction, where you only have a binary mask as ground truth (no color/shading info)"

Notes:
The core idea is that rasterization has a binary hard edge over space, and occlusion via the depth buffer is a binary hard edge over depth.
By making both of these soft, it allows differentiation.

Paper:
https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Soft_Rasterizer_A_Differentiable_Renderer_for_Image-Based_3D_Reasoning_ICCV_2019_paper.pdf

Link to this for slightly newer methods of differentiable rasterization:
https://jjbannister.github.io/tinydiffrast/

*/
