#include <hip/hip_runtime.h>

extern "C" __global__ void mandelbrot_kernel(float cr0, float ci0, float zoom,
    unsigned char* pixels, int w, int h) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    // Map pixel to complex plane centered at (cr0, ci0)
    float c_re = cr0 + ((float)x / (w-1) - 0.5f) * zoom;
    float c_im = ci0 + ((float)y / (h-1) - 0.5f) * zoom;

    // Escape-time iteration with smooth/normalized coloring
    float z_re = 0, z_im = 0;
    int iter;

    for (iter = 0; iter < 4096 && (z_re*z_re + z_im*z_im) <= 4.0f; iter++) {
        float tmp = z_re * z_re - z_im * z_im + c_re;
        z_im = 2.0f * z_re * z_im + c_im;
        z_re = tmp;
    }

    int idx = (y * w + x) * 4;

    if (iter == 4096) {
        // Inside the Mandelbrot set — near-black with boundary glow
        unsigned char shade = iter / 256;
        pixels[idx+0] = shade >> 2;
        pixels[idx+1] = shade >> 3;
        pixels[idx+2] = (shade >> 2) + 48;
    } else {
        // Continuous normalized iteration count for smooth gradients
        float log_zn = __logf(z_re*z_re + z_im*z_im) * 0.5f;
        float nu = __logf(log_zn / __logf(2.0f)) / __logf(2.0f);

        // Color palette with varying-frequency cosine cycling
        int t = iter + (int)(nu * 8);

        unsigned char r = ((unsigned char)(127.5f + 127.5f * __sinf(t * 0.06f)));
        unsigned char g = ((unsigned char)(127.5f + 127.5f * __sinf(t * 0.06f + 2.094f)));
        unsigned char b = ((unsigned char)(127.5f + 127.5f * __sinf(t * 0.06f + 4.189f)));

        pixels[idx+0] = r;
        pixels[idx+1] = g;
        pixels[idx+2] = b;
    }
    pixels[idx+3] = 255; // opaque alpha
}
