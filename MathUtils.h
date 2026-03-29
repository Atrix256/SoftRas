#pragma once

struct Point2D
{
    float x, y;
};

struct Point2DI
{
    int x, y;
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
