// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_DENOISER_H
#define PBRT_DENOISER_H

#include <pbrt/pbrt.h>

#include <pbrt/util/color.h>
#include <pbrt/util/vecmath.h>

#if defined(PBRT_WITH_OIDN)
#include <OpenImageDenoise/oidn.hpp>
#endif

namespace pbrt {

class Denoiser {
  public:
    Denoiser(Vector2i resolution, bool haveAlbedoAndNormal);

    // All pointers should be to GPU memory.
    // |n| and |albedo| should be nullptr iff \haveAlbedoAndNormal| is false.
    virtual void Denoise(RGB *rgb, Normal3f *n, RGB *albedo, RGB *result) = 0;

  protected:
    Vector2i resolution;
    bool haveAlbedoAndNormal;
};

#if defined(PBRT_WITH_OIDN)
class OIDNDenoiser: public Denoiser {
  public:
    OIDNDenoiser(Vector2i resolution, bool haveAlbedoAndNormal);

    // All pointers should be to GPU memory.
    // |n| and |albedo| should be nullptr iff \haveAlbedoAndNormal| is false.
    void Denoise(RGB *rgb, Normal3f *n, RGB *albedo, RGB *result);

  private:
    bool filterFeatures {false};

    oidn::DeviceRef oidnDevice;
    
    oidn::BufferRef bufferColor;
    oidn::BufferRef bufferColorOutput;

    oidn::BufferRef bufferAlbedo;
    oidn::BufferRef bufferAlbedoOutput;

    oidn::BufferRef bufferNormal;
    oidn::BufferRef bufferNormalOutput;

    oidn::FilterRef oidnAlbedoFilter;
    oidn::FilterRef oidnNormalFilter;
    oidn::FilterRef oidnColorFilter;
};
#endif



}  // namespace pbrt

#endif  // PBRT_GPU_DENOISER_H
