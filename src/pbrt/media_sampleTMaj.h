// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// Modifications Copyright 2023 Intel Corporation.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_V4_MEDIA_SAMPLETMAJ_H
#define PBRT_V4_MEDIA_SAMPLETMAJ_H

#include <pbrt/pbrt.h>

#include <pbrt/base/medium.h>
#include <pbrt/interaction.h>
#include <pbrt/paramdict.h>
#include <pbrt/textures.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/error.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/scattering.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/transform.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/SampleFromVoxels.h>
#ifdef PBRT_BUILD_GPU_RENDERER
#include <nanovdb/util/CudaDeviceBuffer.h>
#endif  // PBRT_BUILD_GPU_RENDERER

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

namespace pbrt {

template <typename F>
PBRT_CPU_GPU SampledSpectrum SampleT_maj(Ray ray, Float tMax, Float u, RNG &rng,
                                         const SampledWavelengths &lambda, F callback) {
    auto sample = [&](auto medium) {
        using M = typename std::remove_reference_t<decltype(*medium)>;
        return SampleT_maj<M>(ray, tMax, u, rng, lambda, callback);
    };
    return ray.medium.Dispatch(sample);
}

template <typename ConcreteMedium, typename F>
PBRT_CPU_GPU SampledSpectrum SampleT_maj(Ray ray, Float tMax, Float u, RNG &rng,
                                         const SampledWavelengths &lambda, F callback) {
    // Normalize ray direction and update _tMax_ accordingly
    tMax *= Length(ray.d);
    ray.d = Normalize(ray.d);

    // Initialize _MajorantIterator_ for ray majorant sampling
    ConcreteMedium *medium = ray.medium.Cast<ConcreteMedium>();
    typename ConcreteMedium::MajorantIterator iter = medium->SampleRay(ray, tMax, lambda);

    // Generate ray majorant samples until termination
    SampledSpectrum T_maj(1.f);
    bool done = false;
    while (!done) {
        // Get next majorant segment from iterator and sample it
        pstd::optional<RayMajorantSegment> seg = iter.Next();
        if (!seg)
            return T_maj;
        // Handle zero-valued majorant for current segment
        if (seg->sigma_maj[lambda.ChannelIdx()] == 0) {
            Float dt = seg->tMax - seg->tMin;
            // Handle infinite _dt_ for ray majorant segment
            if (IsInf(dt))
                dt = std::numeric_limits<Float>::max();

            T_maj *= FastExp(-dt * seg->sigma_maj);
            continue;
        }

        // Generate samples along current majorant segment
        Float tMin = seg->tMin;
        while (true) {
            // Try to generate sample along current majorant segment
            Float t = tMin + SampleExponential(u, seg->sigma_maj[lambda.ChannelIdx()]);
            PBRT_DBG("Sampled t = %f from tMin %f u %f sigma_maj[%d] %f\n", t, tMin, u,
                     lambda.ChannelIdx(), seg->sigma_maj[lambda.ChannelIdx()]);
            u = rng.Uniform<Float>();
            if (t < seg->tMax) {
                // Call callback function for sample within segment
                PBRT_DBG("t < seg->tMax\n");
                T_maj *= FastExp(-(t - tMin) * seg->sigma_maj);
                MediumProperties mp = medium->SamplePoint(ray(t), lambda);
                if (!callback(ray(t), mp, seg->sigma_maj, T_maj)) {
                    // Returning out of doubly-nested while loop is not as good perf. wise
                    // on the GPU vs using "done" here.
                    done = true;
                    break;
                }
                T_maj = SampledSpectrum(1.f);
                tMin = t;

            } else {
                // Handle sample past end of majorant segment
                Float dt = seg->tMax - tMin;
                // Handle infinite _dt_ for ray majorant segment
                if (IsInf(dt))
                    dt = std::numeric_limits<Float>::max();

                T_maj *= FastExp(-dt * seg->sigma_maj);
                PBRT_DBG("Past end, added dt %f * maj[%d] %f\n", dt, lambda.ChannelIdx(), seg->sigma_maj[lambda.ChannelIdx()]);
                break;
            }
        }
    }
    return SampledSpectrum(1.f);
}

template <typename F>
PBRT_CPU_GPU SampledSpectrum SampleT_maj_resampling(Ray ray, Float tMax, Float u, RNG &rng,
                                                    const SampledWavelengths &lambda,
                                                    bool &passThroughMedium,
                                                    bool guideScatterDecision, Float volScatterProb,
                                                    Float &volumeRatioZeroCandidateCompensation,
                                                    Float &majorantScale,
                                                    F callback) {
    auto sample = [&](auto medium) {
        using M = typename std::remove_reference_t<decltype(*medium)>;
        return SampleT_maj_resampling<M>(ray, tMax, u, rng, lambda, passThroughMedium, guideScatterDecision, volScatterProb, volumeRatioZeroCandidateCompensation, majorantScale, callback);
    };
    return ray.medium.Dispatch(sample);
}

template <typename ConcreteMedium, typename F>
PBRT_CPU_GPU SampledSpectrum SampleT_maj_resampling(Ray ray, Float tMax, Float u, RNG &rng,
                                                    const SampledWavelengths &lambda,
                                                    bool &passThroughMedium,
                                                    bool guideScatterDecision, Float volScatterProb,
                                                    Float &volumeRatioZeroCandidateCompensation,
                                                    Float &majorantScale,
                                                    F callback) {
    // Normalize ray direction and update _tMax_ accordingly
    tMax *= Length(ray.d);
    ray.d = Normalize(ray.d);

    // Initialize _MajorantIterator_ for ray majorant sampling
    ConcreteMedium *medium = ray.medium.Cast<ConcreteMedium>();
    typename ConcreteMedium::MajorantIterator iter = medium->SampleRay(ray, tMax, lambda);

    auto iter_preprocess = iter;
    int totalSegmentCount = 0;
    Float totalLength = 0.f;
    while(true) {
        pstd::optional<RayMajorantSegment> seg = iter_preprocess.Next();
        if (!seg)
            break;

        if (seg->sigma_maj[lambda.ChannelIdx()] == 0)
            continue;

        totalSegmentCount ++;
        totalLength += seg->sigma_maj[lambda.ChannelIdx()] * (seg->tMax - seg->tMin);
    }

    if (totalSegmentCount == 0) {
        passThroughMedium = false;
        return SampledSpectrum(1.f);
    }
    passThroughMedium = true;

    majorantScale = 1.0f;
    volumeRatioZeroCandidateCompensation = volScatterProb;
    if (guideScatterDecision) {
        Float minTotalLength = -std::log(1 - volScatterProb);
        if (minTotalLength > totalLength) {
            majorantScale = minTotalLength / totalLength;
            totalLength = minTotalLength;
        }
        Float expNegTotalLength = FastExp(-totalLength);
        volumeRatioZeroCandidateCompensation = volScatterProb / (1 - expNegTotalLength);
    }

    // Generate ray majorant samples until termination
    SampledSpectrum T_maj(1.f);
    bool done = false;
    while (!done) {
        // Get next majorant segment from iterator and sample it
        pstd::optional<RayMajorantSegment> seg = iter.Next();
        if (!seg)
            return T_maj;
        // Handle zero-valued majorant for current segment
        if (seg->sigma_maj[lambda.ChannelIdx()] == 0) {
            Float dt = seg->tMax - seg->tMin;
            // Handle infinite _dt_ for ray majorant segment
            if (IsInf(dt))
                dt = std::numeric_limits<Float>::max();

            T_maj *= FastExp(-dt * seg->sigma_maj);
            continue;
        }

        // Generate samples along current majorant segment
        Float tMin = seg->tMin;
        while (true) {
            // Try to generate sample along current majorant segment
            Float t = tMin + SampleExponential(u, seg->sigma_maj[lambda.ChannelIdx()]);
            PBRT_DBG("Sampled t = %f from tMin %f u %f sigma_maj[%d] %f\n", t, tMin, u,
                     lambda.ChannelIdx(), seg->sigma_maj[lambda.ChannelIdx()]);
            u = rng.Uniform<Float>();
            if (t < seg->tMax) {
                // Call callback function for sample within segment
                PBRT_DBG("t < seg->tMax\n");
                T_maj *= FastExp(-(t - tMin) * seg->sigma_maj);
                MediumProperties mp = medium->SamplePoint(ray(t), lambda);
                if (!callback(ray(t), mp, seg->sigma_maj, T_maj)) {
                    // Returning out of doubly-nested while loop is not as good perf. wise
                    // on the GPU vs using "done" here.
                    done = true;
                    break;
                }
                T_maj = SampledSpectrum(1.f);
                tMin = t;

            } else {
                // Handle sample past end of majorant segment
                Float dt = seg->tMax - tMin;
                // Handle infinite _dt_ for ray majorant segment
                if (IsInf(dt))
                    dt = std::numeric_limits<Float>::max();

                T_maj *= FastExp(-dt * seg->sigma_maj);
                PBRT_DBG("Past end, added dt %f * maj[%d] %f\n", dt, lambda.ChannelIdx(), seg->sigma_maj[lambda.ChannelIdx()]);
                break;
            }
        }
    }
    return SampledSpectrum(1.f);
}

}  // namespace pbrt

#endif  // PBRT_MEDIA_H
