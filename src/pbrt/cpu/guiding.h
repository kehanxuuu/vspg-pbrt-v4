/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2013 Intel Corporation. */

#ifndef PBRT_GUIDING_H
#define PBRT_GUIDING_H


#include <pbrt/interaction.h>
#include <openpgl/cpp/OpenPGL.h>

#include <iostream>

namespace pbrt {


struct GuidedBSDF{

    GuidedBSDF(openpgl::cpp::Field* guiding_field,
    openpgl::cpp::SurfaceSamplingDistribution* surfaceSamplingDistribution, bool enableGuiding = true){
        m_guiding_field = guiding_field;
        m_surfaceSamplingDistribution = surfaceSamplingDistribution;
        m_enableGuiding = enableGuiding;
    }

    bool init(const BSDF* bsdf, const RayDifferential& ray, pstd::optional<pbrt::ShapeIntersection> si, float &rand){
        m_bsdf = bsdf;
        const Point3f p = ray.o + si->tHit * ray.d;
        pgl_point3f pglP = openpgl::cpp::Point3(p[0], p[1], p[2]);
        bool success = false;

        if (IsNonSpecular(bsdf->Flags())) {
            if(m_surfaceSamplingDistribution->Init(m_guiding_field, pglP, rand)){
                Normal3f n = si->intr.shading.n;
                pgl_point3f pglN = openpgl::cpp::Vector3(n[0], n[1], n[2]);
                m_surfaceSamplingDistribution->ApplyCosineProduct(pglN);
                success = true;
                //std::cout << "init: success" << std::endl;
            }
        } else {
            success = false;
        }
        useGuiding = m_enableGuiding ? success : false;
        return success;
    }

    BxDFFlags Flags() const { 
        return m_bsdf->Flags(); 
    }

    SampledSpectrum f(Vector3f woRender, Vector3f wiRender,
                                    TransportMode mode = TransportMode::Radiance) const {
        return m_bsdf->f(woRender, wiRender, mode);
    }

    pstd::optional<BSDFSample> Sample_f(
        Vector3f woRender, Float u, Point2f u2, Float v,
        TransportMode mode = TransportMode::Radiance,
        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {

            pstd::optional<BSDFSample> bs = {};
            bool sampleBSDF = true;
            if (useGuiding) {
                if(guidingProbability > v) {
                    //u /= guidingProbability;
                    //u /= (1.0f - guidingProbability);
                    sampleBSDF = false;
                } else {
                    //u -= guidingProbability;
                    //u /= (1.0f - guidingProbability);
                    sampleBSDF = true;
                }
            }

            if (sampleBSDF){
                bs = m_bsdf->Sample_f(woRender, u, u2, mode, sampleFlags);
                if(bs && IsNonSpecular(m_bsdf->Flags())) {
                    //bs->eta = 1.0f;
                    //bs->sampledRoughness = 1.0f;
                    //if (IsNonSpecular(bsdf->Flags())) {
                    bs->f = m_bsdf->f(woRender, bs->wi, mode);
                    bs->pdf = m_bsdf->PDF(woRender, bs->wi);
                    //bs->pdfIsProportional = false;
                    //bs->flags = m_bsdf->Flags();
                    //}
                }
                if(bs && useGuiding) {
                    pgl_vec3f pglwi = openpgl::cpp::Vector3(bs->wi[0], bs->wi[1], bs->wi[2]);
                    float guidedPDF = m_surfaceSamplingDistribution->PDF(pglwi);
                    bs->pdf = ((1.0f - guidingProbability) * bs->pdf) + (guidingProbability * guidedPDF); 
                }
            } else {
                pgl_point2f sample2D = openpgl::cpp::Point2(u2[0], u2[1]);
                pgl_vec3f pglwi;
                float guidedPDF = m_surfaceSamplingDistribution->SamplePDF(sample2D, pglwi);
                
                Vector3f wiRender = Vector3f(pglwi.x, pglwi.y, pglwi.z);
                SampledSpectrum f = m_bsdf->f(woRender, wiRender, mode);
                Float bsdfPDF = m_bsdf->PDF(woRender, wiRender);
                if(bsdfPDF > 0.f) {
                    BxDFFlags flags = m_bsdf->Flags();
                    Float sampledRoughness = m_bsdf->GetRoughness();
                    Float eta = m_bsdf->GetEta();
                    bool pdfIsProportional = false;

                    float pdf = ((1.0f - guidingProbability) * bsdfPDF) + (guidingProbability * guidedPDF); 
                    bs = BSDFSample(f, wiRender, pdf, flags, sampledRoughness, eta, pdfIsProportional);
                }
            }
            return bs;

    }

    float PDF(Vector3f woRender, Vector3f wiRender,
              TransportMode mode = TransportMode::Radiance,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        
        float bsdfPDF = m_bsdf->PDF(woRender, wiRender); 
        if (useGuiding){
            pgl_vec3f pglwi = openpgl::cpp::Vector3(wiRender[0], wiRender[1], wiRender[2]);
            float guidedPDF = m_surfaceSamplingDistribution->PDF(pglwi);
            return ((1.0f - guidingProbability) * bsdfPDF) + (guidingProbability * guidedPDF);
        } else {
            return bsdfPDF;
        }
    }

private:
    bool m_enableGuiding = true;
    float guidingProbability = 0.5f;
    bool useGuiding = false;

    openpgl::cpp::Field* m_guiding_field;
    openpgl::cpp::SurfaceSamplingDistribution* m_surfaceSamplingDistribution;
    const BSDF* m_bsdf;
    //Sampler* m_sampler;
};

inline openpgl::cpp::PathSegment* guiding_newSurfacePathSegment(openpgl::cpp::PathSegmentStorage* pathSegmentStorage, const RayDifferential& ray, pstd::optional<pbrt::ShapeIntersection> si)
{
    const SurfaceInteraction &isect = si->intr;
    
    const pgl_vec3f pglZero = openpgl::cpp::Vector3(0.0f, 0.0f, 0.0f);
    const pgl_vec3f pglOne = openpgl::cpp::Vector3(1.0f, 1.0f, 1.0f);
    
    const Vector3f wo = -ray.d;
    const Point3f p = ray.o + si->tHit * ray.d;
    const Normal3f n = isect.shading.n;
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
        const pgl_vec3f pglLd = openpgl::cpp::Vector3(std::max(0.f, LdRGB.r), std::max(0.f, LdRGB.g), std::max(0.f, LdRGB.b));
        openpgl::cpp::AddScatteredContribution(pathSegmentData, pglLd);
    }
}

inline void guiding_addSurfaceEmission(openpgl::cpp::PathSegment* pathSegmentData, const SampledSpectrum& Le, float misWeight, SampledWavelengths &lambda, const RGBColorSpace *colorSpace)
{
    if(pathSegmentData) {
        const RGB LeRGB = Le.ToRGB(lambda, *colorSpace);
        const pgl_vec3f pglLe = openpgl::cpp::Vector3(std::max(0.f, LeRGB.r), std::max(0.f, LeRGB.g), std::max(0.f, LeRGB.b));
        openpgl::cpp::SetDirectContribution(pathSegmentData, pglLe);
        openpgl::cpp::SetMiWeight(pathSegmentData, misWeight);
    }
}

inline void guiding_addInfiniteLightEmission(openpgl::cpp::PathSegmentStorage* pathSegmentStorage, float guidingInfiniteLightDistance, const RayDifferential& ray, const SampledSpectrum& Le, float misWeight, SampledWavelengths &lambda, const RGBColorSpace *colorSpace)
{
    const pgl_vec3f pglZero = openpgl::cpp::Vector3(0.f, 0.f, 0.f);
    const pgl_vec3f pglOne = openpgl::cpp::Vector3(1.f, 1.f, 1.f);
    
    openpgl::cpp::PathSegment* pathSegmentData = pathSegmentStorage->NextSegment();
    if(pathSegmentData) {
        const RGB LeRGB = Le.ToRGB(lambda, *colorSpace);
        const pgl_vec3f pglLe = openpgl::cpp::Vector3(std::max(0.f, LeRGB.r), std::max(0.f, LeRGB.g), std::max(0.f, LeRGB.b));
        

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

        openpgl::cpp::SetScatteredContribution(pathSegmentData, pglZero);
    }
}

inline void guiding_addSurfaceData(openpgl::cpp::PathSegment* pathSegmentData, const SampledSpectrum& bsdfWeight, const Vector3f& wi, const Float eta, const Float sampledRoughness, const Float bsdfPDF, const Float survivalProbability, SampledWavelengths &lambda, const RGBColorSpace *colorSpace)
{
    const pgl_vec3f pglZero = openpgl::cpp::Vector3(0.0f, 0.0f, 0.0f);
    const pgl_vec3f pglOne = openpgl::cpp::Vector3(1.0f, 1.0f, 1.0f);
    
    if(pathSegmentData) {
        bool is_delta = sampledRoughness < 0.001f;
        const RGB bsdfWeightRGB = bsdfWeight.ToRGB(lambda, *colorSpace);
        const pgl_vec3f pglBsdfWeight = openpgl::cpp::Vector3(std::max(0.f, bsdfWeightRGB.r), std::max(0.f, bsdfWeightRGB.g), std::max(0.f, bsdfWeightRGB.b));
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
