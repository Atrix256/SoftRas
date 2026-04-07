#pragma once

#include <cmath>
#include <array>

template <size_t N>
using Vec = std::array<float, N>;

using Vec2 = Vec<2>;
using Vec3 = Vec<3>;
using Vec4 = Vec<4>;

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
inline Vec<N> Clamp(const Vec<N>& A, float theMin, float theMax)
{
    Vec<N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = Clamp(A[i], theMin, theMax);
    return ret;
}

// Vec / Vec

template <size_t N>
inline float Dot(const Vec<N>& A, const Vec<N>& B)
{
    float result = 0.0f;
    for (size_t i = 0; i < N; ++i)
        result += A[i] * B[i];
    return result;
}

template <size_t N>
inline Vec<N> operator+ (const Vec<N>& A, const Vec<N>& B)
{
    Vec<N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A[i] + B[i];
    return ret;
}

template <size_t N>
inline Vec<N> operator- (const Vec<N>& A, const Vec<N>& B)
{
    Vec<N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A[i] - B[i];
    return ret;
}

template <size_t N>
inline Vec<N> operator* (const Vec<N>& A, const Vec<N>& B)
{
    Vec<N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A[i] * B[i];
    return ret;
}

template <size_t N>
inline Vec<N> operator/ (const Vec<N>& A, const Vec<N>& B)
{
    Vec<N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A[i] / B[i];
    return ret;
}

// Vec / Scalar

template <size_t N>
inline Vec<N> operator* (const Vec<N>& A, float B)
{
    Vec<N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A[i] * B;
    return ret;
}

template <size_t N>
inline Vec<N> operator+ (const Vec<N>& A, float B)
{
    Vec<N> ret;
    for (size_t i = 0; i < N; ++i)
        ret[i] = A[i] + B;
    return ret;
}

// Other

namespace std
{
    template <size_t N>
    inline Vec<N> min(const Vec<N>& A, const Vec<N>& B)
    {
        Vec<N> ret;
        for (size_t i = 0; i < N; ++i)
            ret[i] = std::min(A[i], B[i]);
        return ret;
    }
};
