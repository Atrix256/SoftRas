#pragma once

#include <cmath>
#include <array>

static const float c_pi = 3.14159265359f;

template <size_t N>
using Vec = std::array<float, N>;

template <size_t M, size_t N>
using Mat = std::array<std::array<float, N>, M>;

using Vec2 = Vec<2>;
using Vec3 = Vec<3>;
using Vec4 = Vec<4>;

using Mat4x4 = Mat<4, 4>;

// Returns 2 times the signed triangle area. The result is positive if
// abc is ccw, negative if abc is cw, zero if abc is degenerate.
inline float Signed2DTriArea(const Vec2& a, const Vec2& b, const Vec2& c)
{
    return (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]);
}

inline float Sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

inline float Sign(float f)
{
    if (f < 0.0f)
        return -1.0f;
    else if (f > 0.0f)
        return 1.0f;
    else
        return 0.0f;
}

template <typename T>
T Clamp(T value, T theMin, T theMax)
{
    if (value <= theMin)
        return theMin;
    else if (value >= theMax)
        return theMax;
    else
        return value;
}

template <size_t N>
Vec<N> Clamp(const Vec<N>& A, float theMin, float theMax)
{
    Vec<N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = Clamp(A[i], theMin, theMax);
    return ret;
}

template <size_t N>
Vec2 XY(const Vec<N>& A)
{
    static_assert(N >= 2, "Vec must have at least 2 components");
    return Vec2{ A[0], A[1] };
}

template <size_t N>
Vec3 XYZ(const Vec<N>& A)
{
    static_assert(N >= 3, "Vec must have at least 3 components");
    return Vec3{ A[0], A[1], A[2] };
}

// ======================================= Vec / Vec

template <size_t N>
float Dot(const Vec<N>& A, const Vec<N>& B)
{
    float result = 0.0f;
    for (size_t i = 0; i < N; ++i)
        result += A[i] * B[i];
    return result;
}

template <size_t N>
float LenSquared(const Vec<N>& A)
{
    return Dot(A, A);
}

template <size_t N>
float Len(const Vec<N>& A)
{
    return std::sqrt(LenSquared(A));
}

template <size_t N>
Vec<N> operator+ (const Vec<N>& A, const Vec<N>& B)
{
    Vec<N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A[i] + B[i];
    return ret;
}

template <size_t N>
Vec<N> operator- (const Vec<N>& A, const Vec<N>& B)
{
    Vec<N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A[i] - B[i];
    return ret;
}

template <size_t N>
Vec<N> operator* (const Vec<N>& A, const Vec<N>& B)
{
    Vec<N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A[i] * B[i];
    return ret;
}

template <size_t N>
Vec<N> operator/ (const Vec<N>& A, const Vec<N>& B)
{
    Vec<N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A[i] / B[i];
    return ret;
}

template <size_t N>
Vec<N>& operator+= (Vec<N>& A, const Vec<N>& B)
{
    for (size_t i = 0; i < N; ++i)
        A[i] += B[i];
    return A;
}

template <size_t N>
Vec<N>& operator-= (Vec<N>& A, const Vec<N>& B)
{
    for (size_t i = 0; i < N; ++i)
        A[i] -= B[i];
    return A;
}

template <size_t N>
Vec<N>& operator*= (Vec<N>& A, const Vec<N>& B)
{
    for (size_t i = 0; i < N; ++i)
        A[i] *= B[i];
    return A;
}

template <size_t N>
Vec<N>& operator/= (Vec<N>& A, const Vec<N>& B)
{
    for (size_t i = 0; i < N; ++i)
        A[i] /= B[i];
    return A;
}

namespace std
{
    template <size_t N>
    Vec<N> min(const Vec<N>& A, const Vec<N>& B)
    {
        Vec<N> ret;
        for (size_t i = 0; i < N; ++i)
            ret[i] = std::min(A[i], B[i]);
        return ret;
    }

    template <size_t N>
    Vec<N> abs(const Vec<N>& A)
    {
        Vec<N> ret;
        for (size_t i = 0; i < N; ++i)
            ret[i] = std::abs(A[i]);
        return ret;
    }

    template <size_t N>
    Vec<N> pow(const Vec<N>& A, const Vec<N>& B)
    {
        Vec<N> ret;
        for (size_t i = 0; i < N; ++i)
            ret[i] = std::pow(A[i], B[i]);
        return ret;
    }
};

// ======================================= Vec / Scalar

template <size_t N>
Vec<N> operator+ (const Vec<N>& A, float B)
{
    Vec<N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A[i] + B;
    return ret;
}

template <size_t N>
Vec<N> operator- (const Vec<N>& A, float B)
{
    Vec<N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A[i] - B;
    return ret;
}

template <size_t N>
Vec<N> operator* (const Vec<N>& A, float B)
{
    Vec<N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A[i] * B;
    return ret;
}

template <size_t N>
Vec<N> operator/ (const Vec<N>& A, float B)
{
    Vec<N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A[i] / B;
    return ret;
}

template <size_t N>
Vec<N> operator+ (float A, const Vec<N>& B)
{
    Vec<N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A + B[i];
    return ret;
}

template <size_t N>
Vec<N> operator- (float A, const Vec<N>& B)
{
    Vec<N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A - B[i];
    return ret;
}

template <size_t N>
Vec<N> operator* (float A, const Vec<N>& B)
{
    Vec<N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A * B[i];
    return ret;
}

template <size_t N>
Vec<N> operator/ (float A, const Vec<N>& B)
{
    Vec<N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A / B[i];
    return ret;
}

template <size_t N>
Vec<N>& operator+= (Vec<N>& A, float B)
{
    for (size_t i = 0; i < N; ++i)
        A[i] += B;
    return A;
}

template <size_t N>
Vec<N>& operator-= (Vec<N>& A, float B)
{
    for (size_t i = 0; i < N; ++i)
        A[i] -= B;
    return A;
}

template <size_t N>
Vec<N>& operator*= (Vec<N>& A, float B)
{
    for (size_t i = 0; i < N; ++i)
        A[i] *= B;
    return A;
}

template <size_t N>
Vec<N>& operator/= (Vec<N>& A, float B)
{
    for (size_t i = 0; i < N; ++i)
        A[i] /= B;
    return A;
}

// ======================================= Vec / Matrix

template <size_t N>
Vec<N> MatMul(const Vec<N>& A, const Mat<N, N>& B)
{
    Vec<N> ret;
    for (int i = 0; i < N; ++i)
    {
        ret[i] = 0.0f;
        for (int j = 0; j < N; ++j)
            ret[i] += A[j] * B[i][j];
    }
    return ret;
}

// ======================================= Matrix

inline Mat4x4 PerspectiveFovLH_ReverseZ_InfiniteDepth(float FovAngleYDegrees, float AspectRatio, float NearZ, bool leftHanded)
{
    float FovAngleY = FovAngleYDegrees * c_pi / 180.0f;

    float SinFov = sin(0.5f * FovAngleY);
    float CosFov = cos(0.5f * FovAngleY);

    float Height = CosFov / SinFov;
    float Width = Height / AspectRatio;

    Mat4x4 ret;
    ret[0][0] = Width;
    ret[0][1] = 0.0f;
    ret[0][2] = 0.0f;
    ret[0][3] = 0.0f;

    ret[1][0] = 0.0f;
    ret[1][1] = Height;
    ret[1][2] = 0.0f;
    ret[1][3] = 0.0f;

    ret[2][0] = 0.0f;
    ret[2][1] = 0.0f;
    ret[2][2] = 0.0f;
    ret[2][3] = leftHanded ? 1.0f : -1.0f;

    ret[3][0] = 0.0f;
    ret[3][1] = 0.0f;
    ret[3][2] = NearZ;
    ret[3][3] = 0.0f;
    return ret;
}

// ======================================= Graphics
Vec3 LinearToSRGB(const Vec3& linearCol)
{
    Vec3 sRGBLo = linearCol * 12.92f;
    Vec3 sRGBHi = (std::pow(std::abs(linearCol), Vec3{ 1.0f / 2.4f, 1.0f / 2.4f, 1.0f / 2.4f }) * 1.055f) - 0.055f;
    Vec3 sRGB;
    sRGB[0] = linearCol[0] <= 0.0031308f ? sRGBLo[0] : sRGBHi[0];
    sRGB[1] = linearCol[1] <= 0.0031308f ? sRGBLo[1] : sRGBHi[1];
    sRGB[2] = linearCol[2] <= 0.0031308f ? sRGBLo[2] : sRGBHi[2];
    return sRGB;
}

Vec3 SRGBToLinear(const Vec3& sRGBCol)
{
    Vec3 linearRGBLo = sRGBCol / 12.92f;
    Vec3 linearRGBHi = std::pow((sRGBCol + 0.055f) / 1.055f, Vec3{ 2.4f, 2.4f, 2.4f });
    Vec3 linearRGB;
    linearRGB[0] = sRGBCol[0] <= 0.04045f ? linearRGBLo[0] : linearRGBHi[0];
    linearRGB[1] = sRGBCol[1] <= 0.04045f ? linearRGBLo[1] : linearRGBHi[1];
    linearRGB[2] = sRGBCol[2] <= 0.04045f ? linearRGBLo[2] : linearRGBHi[2];
    return linearRGB;
}
