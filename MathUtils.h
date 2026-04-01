#pragma once

struct Point2D
{
    float x, y;
};

struct Point3D
{
    float x, y, z;
};

// Returns 2 times the signed triangle area. The result is positive if
// abc is ccw, negative if abc is cw, zero if abc is degenerate.
inline float Signed2DTriArea(Point2D a, Point2D b, Point2D c)
{
    return (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x);
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

inline float Dot(const Point2D& A, const Point2D& B)
{
    return A.x * B.x + A.y * B.y;
}

inline Point2D operator- (const Point2D& A, const Point2D& B)
{
    Point2D ret;
    ret.x = A.x - B.x;
    ret.y = A.y - B.y;
    return ret;
}

inline Point2D operator* (const Point2D& A, const Point2D& B)
{
    Point2D ret;
    ret.x = A.x * B.x;
    ret.y = A.y * B.y;
    return ret;
}

inline Point2D operator* (const Point2D& A, float B)
{
    Point2D ret;
    ret.x = A.x * B;
    ret.y = A.y * B;
    return ret;
}

namespace std
{
    inline Point2D min(const Point2D& A, const Point2D& B)
    {
        Point2D ret;
        ret.x = min(A.x, B.x);
        ret.y = min(A.y, B.y);
        return ret;
    }
};
