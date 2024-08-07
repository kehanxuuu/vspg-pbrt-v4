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
    int channelIdx = lambda.ChannelIdx();
    
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
        if (seg->sigma_maj[channelIdx] == 0) {
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
            Float t = tMin + SampleExponential(u, seg->sigma_maj[channelIdx]);
            PBRT_DBG("Sampled t = %f from tMin %f u %f sigma_maj[%d] %f\n", t, tMin, u,
                     channelIdx, seg->sigma_maj[channelIdx]);
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
                PBRT_DBG("Past end, added dt %f * maj[%d] %f\n", dt, channelIdx, seg->sigma_maj[channelIdx]);
                break;
            }
        }
    }
    return SampledSpectrum(1.f);
}

template <typename F>
PBRT_CPU_GPU SampledSpectrum SampleT_maj_Resampling(Ray ray, Float tMax, Float u, RNG &rng,
                                                    const SampledWavelengths &lambda,
                                                    bool guideScatterDecision, Float volScatterProb,
                                                    Float &volumeRatioZeroCandidateCompensation,
                                                    Float &majorantScale,
                                                    F callback) {
    auto sample = [&](auto medium) {
        using M = typename std::remove_reference_t<decltype(*medium)>;
        return SampleT_maj_Resampling<M>(ray, tMax, u, rng, lambda, guideScatterDecision, volScatterProb, volumeRatioZeroCandidateCompensation, majorantScale, callback);
    };
    return ray.medium.Dispatch(sample);
}

template <typename ConcreteMedium, typename F>
PBRT_CPU_GPU SampledSpectrum SampleT_maj_Resampling(Ray ray, Float tMax, Float u, RNG &rng,
                                                    const SampledWavelengths &lambda,
                                                    bool guideScatterDecision, Float volScatterProb,
                                                    Float &volumeRatioZeroCandidateCompensation,
                                                    Float &majorantScale,
                                                    F callback) {
    int channelIdx = lambda.ChannelIdx();
    
    // Normalize ray direction and update _tMax_ accordingly
    tMax *= Length(ray.d);
    ray.d = Normalize(ray.d);

    // Initialize _MajorantIterator_ for ray majorant sampling
    ConcreteMedium *medium = ray.medium.Cast<ConcreteMedium>();
    typename ConcreteMedium::MajorantIterator iter = medium->SampleRay(ray, tMax, lambda);

    auto iter_preprocess = iter;
    Float totalLength = 0.f;
    while(true) {
        pstd::optional<RayMajorantSegment> seg = iter_preprocess.Next();
        if (!seg)
            break;

        if (seg->sigma_maj[channelIdx] == 0)
            continue;

        totalLength += seg->sigma_maj[channelIdx] * (seg->tMax - seg->tMin);
    }

    if (totalLength == 0.f) {
        return SampledSpectrum(1.f);
    }

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
    int count = 0;
    while (!done) {
        // Get next majorant segment from iterator and sample it
        pstd::optional<RayMajorantSegment> seg = iter.Next();
        if (!seg)
            return T_maj;

        seg->sigma_maj *= majorantScale;

        // Handle zero-valued majorant for current segment
        if (seg->sigma_maj[channelIdx] == 0) {
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
            count ++;
            // Try to generate sample along current majorant segment
            Float t = tMin + SampleExponential(u, seg->sigma_maj[channelIdx]);
            PBRT_DBG("Sampled t = %f from tMin %f u %f sigma_maj[%d] %f\n", t, tMin, u,
                     channelIdx, seg->sigma_maj[channelIdx]);
            u = rng.Uniform<Float>();
            if (t < seg->tMax) {
                if (count > 10000) {
                    std::cout << "Warning: count = " << count << " is too large, must be buggy somewhere!!" << std::endl;
                    break;
                }
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
                PBRT_DBG("Past end, added dt %f * maj[%d] %f\n", dt, channelIdx, seg->sigma_maj[channelIdx]);
                break;
            }
        }
    }
    return SampledSpectrum(1.f);
}

template <typename F>
PBRT_CPU_GPU SampledSpectrum SampleT_maj_OpticalDepthSpace(Ray ray, Float tMax, Float u, RNG &rng,
                                                           const SampledWavelengths &lambda,
                                                           bool guideScatterDecision, Float vsp, Float vspMISRatio,
                                                           bool VilleminMethod,
                                                           SampledSpectrum &beta_factor,
                                                           SampledSpectrum &r_u_factor,
                                                           F callback) {
    auto sample = [&](auto medium) {
        using M = typename std::remove_reference_t<decltype(*medium)>;
        return SampleT_maj_OpticalDepthSpace<M>(ray, tMax, u, rng, lambda, guideScatterDecision, vsp, vspMISRatio, VilleminMethod, beta_factor, r_u_factor, callback);
    };
    return ray.medium.Dispatch(sample);
}

template <typename ConcreteMedium, typename F>
PBRT_CPU_GPU SampledSpectrum SampleT_maj_OpticalDepthSpace(Ray ray, Float tMax, Float u, RNG &rng,
                                                    const SampledWavelengths &lambda,
                                                    bool guideScatterDecision, Float vsp, Float vspMISRatio,
                                                    bool VilleminMethod,
                                                    SampledSpectrum &beta_factor,
                                                    SampledSpectrum &r_u_factor,
                                                    F callback) {
    // Sample distance in optical depth space, ignore majorant grid bounds

    // VilleminMethod = true: can enable or disable collProbBias
    // VilleminMethod = false: our method, default with collision probability biasing
    //      If homogeneous volume -> stop at the first bounce and automatically get the analytical solution

    int channelIdx = lambda.ChannelIdx();
    
    if (!guideScatterDecision || vspMISRatio == 0.f)
        return SampleT_maj(ray, tMax, u, rng, lambda, callback);

    // Normalize ray direction and update _tMax_ accordingly
    tMax *= Length(ray.d);
    ray.d = Normalize(ray.d);

    // Initialize _MajorantIterator_ for ray majorant sampling
    ConcreteMedium *medium = ray.medium.Cast<ConcreteMedium>();
    typename ConcreteMedium::MajorantIterator iter = medium->SampleRay(ray, tMax, lambda);

    auto iter_preprocess = iter;
    Float t_v = 0.f; // In optical depth space
    while(true) {
        pstd::optional<RayMajorantSegment> seg = iter_preprocess.Next();
        if (!seg)
            break;

        if (std::isinf(seg->tMax)) {
            std::cout << "SampleT_maj_OpticalDepthSpace: seg->tMax is inf! " << ray.o << " " << ray.d << " " << std::endl; // Shouldn't reach here
            return SampleT_maj(ray, tMax, u, rng, lambda, callback);
        }

        if (seg->sigma_maj[channelIdx] == 0)
            continue;

        t_v += seg->sigma_maj[channelIdx] * (seg->tMax - seg->tMin);
    }

    if (t_v == 0.f) {
        return SampledSpectrum(1.f);
    }

    Float OneMinusENegTv = 1.f - FastExp(-t_v);
    Float t_n=-1.f, t_n_current=-1.f;
    if (VilleminMethod) {
        // Dose not support "decreasing" vsp (compared to 1 - majorant transmittance)
        if (vsp < 1 - FastExp(-t_v))
            return SampleT_maj(ray, tMax, u, rng, lambda, callback);
        else {
            t_n = -std::log(1.0 - OneMinusENegTv / vsp);
            t_n_current = t_n;
        }
    }

    SampledSpectrum T_maj(1.f), tpScaleFactor(1.f);
    Float t_v_current = t_v;
    Float remainingDist = 0;

    bool deltaTracking = false;
    if (u > vspMISRatio) {
        deltaTracking = true;
        u = (u - vspMISRatio) / (1 - vspMISRatio);
    }
    else {
        u /= vspMISRatio;
    }

    // Generate ray majorant samples until termination
    bool done = false;
    int count = 0;
    bool overTheEnd = false;
    const Float ScatterEpsilon = 1e-5;
    while (!done) {
        // Get next majorant segment from iterator and sample it
        pstd::optional<RayMajorantSegment> seg = iter.Next();
        if (!seg)
            return T_maj;
        // Handle zero-valued majorant for current segment
        if (seg->sigma_maj[channelIdx] == 0 || overTheEnd) {
            Float dt = seg->tMax - seg->tMin;
            // Handle infinite _dt_ for ray majorant segment
            if (IsInf(dt))
                dt = std::numeric_limits<Float>::max();

            T_maj *= FastExp(-dt * seg->sigma_maj);
            continue;
        }

        // Generate samples along current majorant segment
        Float tMin = seg->tMin;
        SampledSpectrum normalizedMaj = seg->sigma_maj / seg->sigma_maj[channelIdx];

        if (remainingDist > 0) {
            tMin += remainingDist / seg->sigma_maj[channelIdx];
            if (tMin > seg->tMax + ScatterEpsilon) {
                Float dist = (seg->tMax - seg->tMin) * seg->sigma_maj[channelIdx];
                t_v_current -= dist;
                t_n_current -= dist;
                remainingDist -= dist;
                T_maj *= FastExp(-(seg->tMax - seg->tMin) * seg->sigma_maj);
                continue;
            }

            t_v_current -= remainingDist;
            t_n_current -= remainingDist;
            remainingDist = 0;
            T_maj *= FastExp(-(tMin - seg->tMin) * seg->sigma_maj);
            MediumProperties mp = medium->SamplePoint(ray(tMin), lambda);

            r_u_factor = SampledSpectrum(vspMISRatio) / tpScaleFactor + SampledSpectrum(1 - vspMISRatio);
            if (!callback(ray(tMin), mp, seg->sigma_maj, T_maj)) {
                // Returning out of doubly-nested while loop is not as good perf. wise
                // on the GPU vs using "done" here.
                break;
            }
            T_maj = SampledSpectrum(1.f);
        }
        
        while (true) {
            count ++;
            // Try to generate sample along current majorant segment
            Float dist = std::numeric_limits<Float>::max();
            SampledSpectrum tpScaleFactorSingleStep;
            if (VilleminMethod) {
                tpScaleFactorSingleStep = 1.0f - FastExp(-t_n_current * normalizedMaj);
                if (!deltaTracking)
                    dist = -std::log(1.0 - u * tpScaleFactorSingleStep[channelIdx]);
            }
            else {
                tpScaleFactorSingleStep = (1.0f - FastExp(-t_v_current * normalizedMaj)) / vsp;
                if (!deltaTracking) {
                    if (u < vsp) {
                        // Sample inside the volume
                        dist = -std::log(1.0 - u * tpScaleFactorSingleStep[channelIdx]);
                    }
                    // Else sample on the surface
                    // No need to set dist because it is already FLOAT_MAX (out of volume boundary)
                }
            }

            if (deltaTracking)
                dist = -std::log(1.0 - u);

            bool passThrough = t_v_current - dist < ScatterEpsilon || dist == 0;
            if (VilleminMethod || !passThrough)
                tpScaleFactor *= tpScaleFactorSingleStep;

            if (passThrough) {
                if (VilleminMethod) {
                    tpScaleFactor /= 1.0f - FastExp(-t_n + t_v);
                }
                else {
                    tpScaleFactor *= FastExp(-t_v_current * normalizedMaj) / (1 - vsp);
                }
                r_u_factor = SampledSpectrum(vspMISRatio) / tpScaleFactor + SampledSpectrum(1 - vspMISRatio);
                overTheEnd = true;
                T_maj *= FastExp(- (seg->tMax - tMin) * seg->sigma_maj);
                break;
            }

            Float t = tMin + dist / seg->sigma_maj[channelIdx];
            
            PBRT_DBG("Sampled t = %f from tMin %f u %f sigma_maj[%d] %f\n", t, tMin, u,
                     channelIdx, seg->sigma_maj[channelIdx]);
            u = rng.Uniform<Float>();
            if (t <= seg->tMax + ScatterEpsilon) {
                if (count > 10000) {
                    std::cout << "Warning: count = " << count << " is too large, must be buggy somewhere!!" << std::endl;
                    std::cout << "Count-related variables: " << dist << " " << t_v_current << " " << std::endl;
                    break;
                }
                t_v_current -= dist;
                t_n_current -= dist;
                remainingDist = 0;
                
                // Call callback function for sample within segment
                PBRT_DBG("t < seg->tMax\n");
                T_maj *= FastExp(-(t - tMin) * seg->sigma_maj);
                MediumProperties mp = medium->SamplePoint(ray(t), lambda);

                r_u_factor = SampledSpectrum(vspMISRatio) / tpScaleFactor + SampledSpectrum(1 - vspMISRatio);
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
                PBRT_DBG("Past end, added dt %f * maj[%d] %f\n", dt, channelIdx, seg->sigma_maj[channelIdx]);

                Float distWithinThisSeg = dt * seg->sigma_maj[channelIdx];
                remainingDist = dist - distWithinThisSeg;
                t_v_current -= distWithinThisSeg;
                t_n_current -= distWithinThisSeg;
                
                break;
            }
        }
    }
    return SampledSpectrum(1.f);
}

}  // namespace pbrt

#endif  // PBRT_MEDIA_H
