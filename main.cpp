#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>

#define THIRTEEN_IMPLEMENTATION
#include "external/thirteen.h"

#define STB_IMAGE_IMPLEMENTATION
#include "external/stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb/stb_image_write.h"

int main(int argc, char** argv)
{
    unsigned char* pixels = Thirteen::Init(800, 600);
    if (!pixels)
        return 1;

    unsigned int frameIndex = 0;

    // Go until window is closed or escape is pressed
    while (Thirteen::Render() && !Thirteen::GetKey(VK_ESCAPE))
    {
        // Write to pixels (RGBA format, 4 bytes per pixel)
        for (int i = 0; i < 800 * 600 * 4; i += 4)
        {
            pixels[i + 0] = 255; // Red
            pixels[i + 1] = unsigned char(frameIndex);   // Green
            pixels[i + 2] = unsigned char(frameIndex / 2);   // Blue
            pixels[i + 3] = 255; // Alpha
        }
        frameIndex++;
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
*/
