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

static const int c_width = 800;
static const int c_height = 600;

static const float c_fovYDegrees = 45.0f;
static const float c_nearPlane = 1.0f;
static const float c_farPlane = 100.0f;
static const bool c_leftHandedProjection = true;

static const Vec3 c_backgroundColor = { 0.2f, 0.2f, 0.2f };

// Constants from equation 1s
static const float c_sigma = 1e-5f;

// Constants from equation 3
static const float c_epsilon = 1e-3f;
static const float c_gamma = 1e-4f;

// Used to keep equations stable. The minimum amount of coverage of a pixel to be considered covered by a triangle.
static const float c_minimumCoverage = 1e-4f;

struct Vertex
{
    Vec3 pos;
    Vec3 color;
    Vec3 normal;
    Vec4 tangent;
    Vec2 UV0;
    Vec2 UV1;
    Vec2 UV2;
    Vec2 UV3;
    int materialID;
    int shapeID;
};

float CrossProduct2D(float x1, float y1, float x2, float y2)
{
    return x1 * y2 - x2 * y1;
}

Vec3 CalculateBarycentricCoordinates(Vec2 A, Vec2 B, Vec2 C, Vec2 P)
{
    Vec3 ret;
    float& u = ret[0];
    float& v = ret[1];
    float& w = ret[2];

    // Vectors for the full triangle
    float ax = B[0] - A[0];
    float ay = B[1] - A[1];
    float bx = C[0] - A[0];
    float by = C[1] - A[1];

    // Vectors for the point P relative to A
    float px = P[0] - A[0];
    float py = P[1] - A[1];

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
float sdTriangle(const Vec2& p, const Vec2& p0, const Vec2& p1, const Vec2& p2)
{
    Vec2 e0 = p1 - p0, e1 = p2 - p1, e2 = p0 - p2;
    Vec2 v0 = p - p0, v1 = p - p1, v2 = p - p2;
    Vec2 pq0 = v0 - e0 * Clamp(Dot(v0, e0) / Dot(e0, e0), 0.0f, 1.0f);
    Vec2 pq1 = v1 - e1 * Clamp(Dot(v1, e1) / Dot(e1, e1), 0.0f, 1.0f);
    Vec2 pq2 = v2 - e2 * Clamp(Dot(v2, e2) / Dot(e2, e2), 0.0f, 1.0f);
    float s = Sign(e0[0] * e2[1] - e0[1] * e2[0]);
    Vec2 d =
        std::min(
            std::min(
                Vec2{ Dot(pq0, pq0), s * (v0[0] * e0[1] - v0[1] * e0[0]) },
                Vec2{ Dot(pq1, pq1), s * (v1[0] * e1[1] - v1[1] * e1[0]) }
            ),
            Vec2{ Dot(pq2, pq2), s * (v2[0] * e2[1] - v2[1] * e2[0]) }
        )
        ;
    return -sqrt(d[0]) * Sign(d[1]);
}

float SoftCoverage(const Vec2& P, const Vec2& A, const Vec2& B, const Vec2& C)
{
    // Equation 1
    float sdf = -sdTriangle(P, A, B, C);
    return Sigmoid(Sign(sdf) * sdf * sdf / c_sigma);
}

Vec2 PixelToClip(int ix, int iy)
{
    // Convert pixel coordinates to normalized device coordinates (NDC)
    float x_ndc = (float(ix) + 0.5f) / float(c_width) * 2.0f - 1.0f;
    float y_ndc = 1.0f - (float(iy) + 0.5f) / float(c_height) * 2.0f; // Invert Y for screen space
    return Vec2{ x_ndc, y_ndc };
}

void RasterizeMesh(unsigned char* pixels, unsigned int width, unsigned int height, const std::vector<Vertex>& mesh, const Mat4x4& viewProjMtx)
{
    // Clear to black
    memset(pixels, 0, width * height * 4);

    // transform the mesh into pixel coordinates
    static std::vector<Vec3> screenPoints;
    screenPoints.resize(mesh.size());
    #if MULTI_THREADED()
    #pragma omp parallel for
    #endif
    for (int index = 0; index < mesh.size(); ++index)
    {
        Vec4 src = Vec4{ mesh[index].pos[0], mesh[index].pos[1], mesh[index].pos[2], 1.0f };
        Vec4 pos = MatMul(src, viewProjMtx);
        pos = pos / pos[3]; // perspective divide

        screenPoints[index][0] = pos[0];
        screenPoints[index][1] = pos[1];

        // It wants normalized linear depth.
        //screenPoints[index][2] = pos[2];
        screenPoints[index][2] = 1.0f - (mesh[index].pos[2] - c_nearPlane) / (c_farPlane - c_nearPlane);
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
            Vec3 color = { 0.0f, 0.0f, 0.0f };
        };
        std::vector<PxInfo> pxInfo;

        for (int ix = 0; ix < (int)width; ++ix)
        {
            pxInfo.clear();
            int pixelIndex = iy * width + ix;

            // If you want to debug a specific pixel, turn off MULTI_THREADED() and click middle mouse button on the pixel to debug.
            if (Thirteen::GetMouseButton(2) && !Thirteen::GetMouseButtonLastFrame(2))
            {
                int mouseX, mouseY;
                Thirteen::GetMousePosition(mouseX, mouseY);
                if (ix == mouseX && iy == mouseY)
                {
                    if (IsDebuggerPresent())
                    {
                        __debugbreak();
                    }
                }
            }

            for (int triangleIndex = 0; triangleIndex < mesh.size() / 3; ++triangleIndex)
            {
                PxInfo newPx;

                const Vertex& vA = mesh[triangleIndex * 3 + 0];
                const Vertex& vB = mesh[triangleIndex * 3 + 1];
                const Vertex& vC = mesh[triangleIndex * 3 + 2];

                const Vec3& sA = screenPoints[triangleIndex * 3 + 0];
                const Vec3& sB = screenPoints[triangleIndex * 3 + 1];
                const Vec3& sC = screenPoints[triangleIndex * 3 + 2];

                const Vec2 resolution = Vec2{ float(width), float(height) };

                Vec2 pxClip = PixelToClip(ix, iy);

                newPx.coverage = SoftCoverage(XY(pxClip), XY(sA), XY(sB), XY(sC));

                if (newPx.coverage < c_minimumCoverage)
                    continue;

                Vec3 uvw = CalculateBarycentricCoordinates(XY(sA), XY(sB), XY(sC), XY(pxClip));

                // The paper says they clamp uvw between 0 and 1 and then they renormalize it to sum to 1
                uvw = Clamp(uvw, 0.0f, 1.0f);
                float sum = uvw[0] + uvw[1] + uvw[2];
                uvw[0] /= sum;
                uvw[1] /= sum;
                uvw[2] /= sum;

                newPx.color = (vA.color * uvw[0] + vB.color * uvw[1] + vC.color * uvw[2]);
                newPx.depth = (sA[2] * uvw[0] + sB[2] * uvw[1] + sC[2] * uvw[2]);

                pxInfo.push_back(newPx);
            }

            Vec3 pixelColor = { 0.0f, 0.0f, 0.0f };
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

            pixels[pixelIndex * 4 + 0] = (unsigned char)Clamp(pixelColor[0] * 255.0f, 0.0f, 255.0f);
            pixels[pixelIndex * 4 + 1] = (unsigned char)Clamp(pixelColor[1] * 255.0f, 0.0f, 255.0f);
            pixels[pixelIndex * 4 + 2] = (unsigned char)Clamp(pixelColor[2] * 255.0f, 0.0f, 255.0f);
            pixels[pixelIndex * 4 + 3] = 255;
        }
    }
}

int main(int argc, char** argv)
{
    std::vector<Vertex> mesh =
    {
        { {0.0f, 0.0f, 5.0f}, {0.0f, 0.0f, 1.0f} },
        { {0.5f, 0.0f, 5.0f}, {0.0f, 0.0f, 1.0f} },
        { {0.5f, 0.5f, 5.0f}, {0.0f, 0.0f, 1.0f} },

        { {0.1f, 0.0f, 5.1f}, {0.0f, 1.0f, 0.0f} },
        { {0.6f, 0.0f, 5.1f}, {0.0f, 1.0f, 0.0f} },
        { {0.6f, 0.5f, 5.1f}, {0.0f, 1.0f, 0.0f} },

        { {0.2f, 0.0f, 5.2f}, {1.0f, 0.0f, 0.0f} },
        { {0.7f, 0.0f, 5.2f}, {1.0f, 0.0f, 0.0f} },
        { {0.7f, 0.5f, 5.2f}, {1.0f, 0.0f, 0.0f} },
    };

    #if 0

    FILE* file = nullptr;
    fopen_s(&file, "Assets/spot/spot_triangulated.bin", "rb");
    if (file)
    {
        fseek(file, 0, SEEK_END);
        size_t fileSize = ftell(file);
        fseek(file, 0, SEEK_SET);
        size_t vertexCount = fileSize / sizeof(Vertex);
        mesh.resize(vertexCount);
        fread(mesh.data(), sizeof(Vertex), vertexCount, file);
        fclose(file);
    }

    // TODO: temp! later, move the camera back
    for (Vertex& v : mesh)
        v.pos[2] += 5.0f;

    #endif

    Mat4x4 viewProjMtx = PerspectiveFovLH_ReverseZ_InfiniteDepth(c_fovYDegrees, float(c_width) / float(c_height), c_nearPlane, c_leftHandedProjection);

    unsigned char* pixels = Thirteen::Init(c_width, c_height);
    if (!pixels)
        return 1;

    while (Thirteen::Render() && !Thirteen::GetKey(VK_ESCAPE))
    {
        RasterizeMesh(pixels, c_width, c_height, mesh, viewProjMtx);

        if (Thirteen::GetKey('V') && !Thirteen::GetKeyLastFrame('V'))
            Thirteen::SetVSync(!Thirteen::GetVSync());
    }

    Thirteen::Shutdown();
    return 0;
}
/*
- why not semi transparent? review paper?
- then gradients
- camera controls


TODO:
- 2d first then 3d
- make optimization work for... vertex positions, material param (color?), lights, camera
- soft ras.
- link to paper and the blog post that talks about more advanced stuff
? should you do the Silhouette aggregate function? "It's used with a separate silhouette loss for unsupervised reconstruction, where you only have a binary mask as ground truth (no color/shading info)"
? perspective correct interpolation of color and depth?

Notes:
The core idea is that rasterization has a binary hard edge over space, and occlusion via the depth buffer is a binary hard edge over depth.
By making both of these soft, it allows differentiation.

Paper:
https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Soft_Rasterizer_A_Differentiable_Renderer_for_Image-Based_3D_Reasoning_ICCV_2019_paper.pdf

Link to this for slightly newer methods of differentiable rasterization:
https://jjbannister.github.io/tinydiffrast/

* paper wants a distance from pixel to closest edge. SDF is exactly that, so i used iq's function (link)
* softmax wants closer triangles to have a larger value, so i used a reversed z infinite depth projection. (did you abandon this? yeah. it wants normalized reverse linear depth)

*/
