/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2013 Intel Corporation. */

#ifndef PBRT_GUIDING_H
#define PBRT_GUIDING_H


#include <pbrt/interaction.h>
#include <openpgl/cpp/OpenPGL.h>

namespace pbrt {

inline openpgl::cpp::PathSegment* guiding_newSurfacePathSegment(openpgl::cpp::PathSegmentStorage* pathSegmentStorage, const RayDifferential& ray, pstd::optional<pbrt::ShapeIntersection> si)
{
    const SurfaceInteraction &isect = si->intr;
    
    const pgl_vec3f pglZero = openpgl::cpp::Vector3(0.0f, 0.0f, 0.0f);
    const pgl_vec3f pglOne = openpgl::cpp::Vector3(1.0f, 1.0f, 1.0f);
    
    const Vector3f wo = -ray.d;
    const Point3f p = ray.o + si->tHit * ray.d;
    const Normal3f n = isect.n;
    pgl_point3f pglP = openpgl::cpp::Point3(p[0], p[1], p[2]);
    pgl_vec3f pglNormal = openpgl::cpp::Vector3(n[0], n[1], n[2]);
    pgl_vec3f pglWo = openpgl::cpp::Vector3(wo[0], wo[1], wo[2]);

    openpgl::cpp::PathSegment* pathSegmentData = pathSegmentStorage->NextSegment();
    openpgl::cpp::SetPosition(pathSegmentData, pglP);
    openpgl::cpp::SetNormal(pathSegmentData, pglNormal);
    openpgl::cpp::SetDirectionOut(pathSegmentData, pglWo);
    openpgl::cpp::SetVolumeScatter(pathSegmentData, false);
    openpgl::cpp::SetScatteredContribution(pathSegmentData, pglZero);
    openpgl::cpp::SetDirectContribution(pathSegmentData, pglZero);
    openpgl::cpp::SetTransmittanceWeight(pathSegmentData, pglOne);
    openpgl::cpp::SetEta(pathSegmentData, 1.0);
    return pathSegmentData;
}

inline void guiding_addScatteredDirectLight(openpgl::cpp::PathSegment* pathSegmentData, const SampledSpectrum& Ld, SampledWavelengths &lambda, const RGBColorSpace *colorSpace)
{
    if(pathSegmentData) {
        const RGB LdRGB = Ld.ToRGB(lambda, *colorSpace);
        const pgl_vec3f pglLd = openpgl::cpp::Vector3(LdRGB.r, LdRGB.g, LdRGB.b);
        openpgl::cpp::AddScatteredContribution(pathSegmentData, pglLd);
    }
}

inline void guiding_addSurfaceEmission(openpgl::cpp::PathSegment* pathSegmentData, const SampledSpectrum& Le, float misWeight, SampledWavelengths &lambda, const RGBColorSpace *colorSpace)
{
    if(pathSegmentData) {
        const RGB LeRGB = Le.ToRGB(lambda, *colorSpace);
        const pgl_vec3f pglLe = openpgl::cpp::Vector3(LeRGB.r, LeRGB.g, LeRGB.b);
        openpgl::cpp::SetDirectContribution(pathSegmentData, pglLe);
        openpgl::cpp::SetMiWeight(pathSegmentData, misWeight);
    }
}

inline void guiding_addInfiniteLightEmission(openpgl::cpp::PathSegmentStorage* pathSegmentStorage, float guidingInfiniteLightDistance, const RayDifferential& ray, const SampledSpectrum& Le, float misWeight, SampledWavelengths &lambda, const RGBColorSpace *colorSpace)
{
    openpgl::cpp::PathSegment* pathSegmentData = pathSegmentStorage->NextSegment();
    if(pathSegmentData) {
        const RGB LeRGB = Le.ToRGB(lambda, *colorSpace);
        const pgl_vec3f pglLe = openpgl::cpp::Vector3(LeRGB.r, LeRGB.g, LeRGB.b);
        

        const Vector3f wo = -ray.d;
        const Point3f p = ray.o + guidingInfiniteLightDistance * ray.d;
        pgl_point3f pglP = openpgl::cpp::Point3(p[0], p[1], p[2]);
        pgl_vec3f pglNormal = openpgl::cpp::Vector3(0.0f, 0.0f, 1.0f);
        pgl_vec3f pglWo = openpgl::cpp::Vector3(wo[0], wo[1], wo[2]);

        openpgl::cpp::SetPosition(pathSegmentData, pglP);
        openpgl::cpp::SetNormal(pathSegmentData, pglNormal);
        openpgl::cpp::SetDirectionOut(pathSegmentData, pglWo);
        
        openpgl::cpp::SetDirectContribution(pathSegmentData, pglLe);
        openpgl::cpp::SetMiWeight(pathSegmentData, misWeight);
    }
}

inline void guiding_addSurfaceData(openpgl::cpp::PathSegment* pathSegmentData, const SampledSpectrum& bsdfWeight, const Vector3f& wi, const Float eta, const Float sampledRoughness, const Float bsdfPDF, const Float survivalProbability, SampledWavelengths &lambda, const RGBColorSpace *colorSpace)
{
    const pgl_vec3f pglZero = openpgl::cpp::Vector3(0.0f, 0.0f, 0.0f);
    const pgl_vec3f pglOne = openpgl::cpp::Vector3(1.0f, 1.0f, 1.0f);
    
    if(pathSegmentData) {
        bool is_delta = sampledRoughness < 0.001f;
        const RGB bsdfWeightRGB = bsdfWeight.ToRGB(lambda, *colorSpace);
        const pgl_vec3f pglBsdfWeight = openpgl::cpp::Vector3(bsdfWeightRGB.r, bsdfWeightRGB.g, bsdfWeightRGB.b);
        const pgl_vec3f pglWi = openpgl::cpp::Vector3(wi[0], wi[1], wi[2]);

        openpgl::cpp::SetTransmittanceWeight(pathSegmentData, pglOne);
        openpgl::cpp::SetVolumeScatter(pathSegmentData, false);
        //openpgl::cpp::SetNormal(pathSegmentData, guiding_vec3f(normal));
        openpgl::cpp::SetDirectionIn(pathSegmentData, pglWi);
        openpgl::cpp::SetPDFDirectionIn(pathSegmentData, bsdfPDF);
        openpgl::cpp::SetScatteringWeight(pathSegmentData, pglBsdfWeight);
        openpgl::cpp::SetIsDelta(pathSegmentData, is_delta);
        openpgl::cpp::SetEta(pathSegmentData, eta);
        openpgl::cpp::SetRoughness(pathSegmentData, sampledRoughness);
        openpgl::cpp::SetRussianRouletteProbability(pathSegmentData, survivalProbability);
    }
}

}  // namespace pbrt

#endif  // PBRT_GUIDING_H
