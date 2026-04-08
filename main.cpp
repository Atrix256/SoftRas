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

// Returns positive values if outside the triangle, negative values if inside the triangle.
// Also returns derivatives and barycentric coordinates.
float sdTriangle(const Vec2& p, const Vec2& A, const Vec2& B, const Vec2& C, Vec2& dDist_dA, Vec2& dDist_dB, Vec2& dDist_dC, Vec3& uvw)
{
    Vec2 edgeAB = B - A;
    Vec2 edgeBC = C - B;
    Vec2 edgeCA = A - C;

    float tAB = Clamp(Dot(p - A, edgeAB) / Dot(edgeAB, edgeAB), 0.0f, 1.0f);
    float tBC = Clamp(Dot(p - B, edgeBC) / Dot(edgeBC, edgeBC), 0.0f, 1.0f);
    float tCA = Clamp(Dot(p - C, edgeCA) / Dot(edgeCA, edgeCA), 0.0f, 1.0f);

    Vec2 closestAB = A + edgeAB * tAB;
    Vec2 closestBC = B + edgeBC * tBC;
    Vec2 closestCA = C + edgeCA * tCA;

    float distSQAB = LenSquared(p - closestAB);
    float distSQBC = LenSquared(p - closestBC);
    float distSQCA = LenSquared(p - closestCA);

    // Calculate barycentric coordinates
    float signedAreaAB = Signed2DTriArea(p, A, B);
    float signedAreaBC = Signed2DTriArea(p, B, C);
    float signedAreaCA = Signed2DTriArea(p, C, A);
    float totalArea = Signed2DTriArea(A, B, C);
    uvw[0] = signedAreaBC / totalArea;
    uvw[1] = signedAreaCA / totalArea;
    uvw[2] = signedAreaAB / totalArea;

    // See if the point is inside the triangle or not
    bool inside = std::abs(Sign(signedAreaAB) + Sign(signedAreaBC) + Sign(signedAreaCA)) == 3.0f;
    float sign = inside ? -1.0f : 1.0f;

    // If AB is the closest edge
    if (distSQAB < distSQBC && distSQAB < distSQCA)
    {
        float dist = std::sqrt(distSQAB) * sign;

        dDist_dA = (1.0f - tAB) * (closestAB - p) / dist;
        dDist_dB = tAB * (closestAB - p) / dist;
        dDist_dC = Vec2{ 0.0f, 0.0f };

        return dist;
    }
    // Else if BC is the closest edge
    else if (distSQBC < distSQCA)
    {
        float dist = std::sqrt(distSQBC) * sign;

        dDist_dA = Vec2{ 0.0f, 0.0f };
        dDist_dB = (1.0f - tBC) * (closestBC - p) / dist;
        dDist_dC = tBC * (closestBC - p) / dist;

        return dist;
    }
    // Else CA is the closest edge
    else
    {
        float dist = std::sqrt(distSQCA) * sign;

        dDist_dA = tCA * (closestCA - p) / dist;
        dDist_dB = Vec2{ 0.0f, 0.0f };
        dDist_dC = (1.0f - tCA) * (closestCA - p) / dist;

        return dist;
    }
}

float SoftCoverage(const Vec2& P, const Vec2& A, const Vec2& B, const Vec2& C, Vec2& dCoverage_dA, Vec2& dCoverage_dB, Vec2& dCoverage_dC, Vec3& uvw)
{
    // sdgTriangle returns positive for outside, and negative inside.
    // The paper wants the opposite of that, so we negate the results.
    Vec2 dDist_dA, dDist_dB, dDist_dC;  
    float sdf = sdTriangle(P, A, B, C, dDist_dA, dDist_dB, dDist_dC, uvw);
    sdf *= -1.0f;
    dDist_dA *= -1.0f;
    dDist_dB *= -1.0f;
    dDist_dC *= -1.0f;

    // Calculate the coverage
    float coverage = Sigmoid(Sign(sdf) * sdf * sdf / c_sigma);

    // Calculate the derivatives
    float dCoverage_dDist = coverage * (1.0f - coverage) * 2.0f * std::abs(sdf) / c_sigma;
    dCoverage_dA = dCoverage_dDist * dDist_dA;
    dCoverage_dB = dCoverage_dDist * dDist_dB;
    dCoverage_dC = dCoverage_dDist * dDist_dC;

    return coverage;
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
    static std::vector<Vec4> screenPoints;
    screenPoints.resize(mesh.size());
    #if MULTI_THREADED()
    #pragma omp parallel for
    #endif
    for (int index = 0; index < mesh.size(); ++index)
    {
        Vec4 src = Vec4{ mesh[index].pos[0], mesh[index].pos[1], mesh[index].pos[2], 1.0f };
        Vec4 pos = MatMul(src, viewProjMtx);

        screenPoints[index][0] = pos[0] / pos[3];
        screenPoints[index][1] = pos[1] / pos[3];

        // It wants normalized negative linear depth. 1 at the near plane, 0 at the far plane.
        //screenPoints[index][2] = pos[2];
        screenPoints[index][2] = 1.0f - (mesh[index].pos[2] - c_nearPlane) / (c_farPlane - c_nearPlane);

        screenPoints[index][3] = 1.0f / pos[3];
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
            Vec2 uv0 = { 0.0f, 0.0f };
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

                const Vec4& sA = screenPoints[triangleIndex * 3 + 0];
                const Vec4& sB = screenPoints[triangleIndex * 3 + 1];
                const Vec4& sC = screenPoints[triangleIndex * 3 + 2];

                const Vec2 resolution = Vec2{ float(width), float(height) };

                Vec2 pxClip = PixelToClip(ix, iy);

                Vec2 dCoverage_dA, dCoverage_dB, dCoverage_dC;
                Vec3 uvw;
                newPx.coverage = SoftCoverage(XY(pxClip), XY(sA), XY(sB), XY(sC), dCoverage_dA, dCoverage_dB, dCoverage_dC, uvw);

                if (newPx.coverage < c_minimumCoverage)
                    continue;

                // The paper says they clamp uvw between 0 and 1 and then they renormalize it to sum to 1
                uvw = Clamp(uvw, 0.0f, 1.0f);
                float sum = uvw[0] + uvw[1] + uvw[2];
                uvw[0] /= sum;
                uvw[1] /= sum;
                uvw[2] /= sum;

                // Screen space barycentric interpolation.
                newPx.depth = (sA[2] * uvw[0] + sB[2] * uvw[1] + sC[2] * uvw[2]);

                // Perspective correct barycentric interpolation.
                float ooW = (sA[3] * uvw[0] + sB[3] * uvw[1] + sC[3] * uvw[2]);
                newPx.color = (vA.color * uvw[0] * sA[3] + vB.color * uvw[1] * sB[3] + vC.color * uvw[2] * sC[3]) / ooW;
                newPx.uv0 = (vA.UV0 * uvw[0] * sA[3] + vB.UV0 * uvw[1] * sB[3] + vC.UV0 * uvw[2] * sC[3]) / ooW;

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

                // Weights - the softmax_max cancels in numerator/denominator
                for (const PxInfo& entry : pxInfo)
                {
                    float weight = entry.coverage * std::exp(entry.depth / c_gamma - softmax_max) / weightDenom;

                    // TODO: whatever shading
                    //pixelColor = pixelColor + Vec3{entry.uv0[0], entry.uv0[1], 0.0f} * weight;
                    pixelColor = pixelColor + entry.color * weight;

                    totalWeight += weight;
                }
            }

            float backgroundWeight = 1.0f - totalWeight;
            pixelColor = pixelColor + c_backgroundColor * backgroundWeight;

            // Convert linear to sRGB
            pixelColor = LinearToSRGB(pixelColor);

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
- gradients
- camera controls


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

* paper wants a distance from pixel to closest edge. SDF is exactly that, so i used iq's function (link)
* softmax wants closer triangles to have a larger value, so i used a reversed z infinite depth projection. (did you abandon this? yeah. it wants normalized reverse linear depth)

*/
