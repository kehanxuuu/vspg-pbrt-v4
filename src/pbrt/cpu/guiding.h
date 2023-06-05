/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2013 Intel Corporation. */

#ifndef PBRT_GUIDING_H
#define PBRT_GUIDING_H


#include <pbrt/interaction.h>
#include <openpgl/cpp/OpenPGL.h>

#include <iostream>

namespace pbrt {


struct GuidedBSDF{

    enum BSDFGuidingType{
        EBSDFGuideMIS = 0,
        EBSDFGuideRIS
    };

    struct RISSample{
        float bsdfPDF {0.f};
        float guidingPDF {0.f};
        float misPDF {0.f};
        float incomingRadiancePDF {0.f};
        SampledSpectrum f;
        Vector3f wiRender;
        float eta {1.0f};
        float sampledRoughness{1.0f};
        BxDFFlags flags;

        float risWeight{0.f};
    };

    GuidedBSDF(Sampler* sampler, openpgl::cpp::Field* guiding_field,
    openpgl::cpp::SurfaceSamplingDistribution* surfaceSamplingDistribution, bool enableGuiding = true){
        m_guiding_field = guiding_field;
        m_surfaceSamplingDistribution = surfaceSamplingDistribution;
        m_enableGuiding = enableGuiding;
        m_sampler = sampler;
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

    pstd::optional<BSDFSample> Sample_f_MIS(
        Vector3f woRender, Float u, Point2f u2,
        TransportMode mode = TransportMode::Radiance,
        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {

        pstd::optional<BSDFSample> bs = {};
        bool sampleBSDF = true;
        if (useGuiding) {
            if(guidingProbability > u) {
                u /= guidingProbability;
                sampleBSDF = false;
            } else {
                u -= guidingProbability;
                u /= (1.0f - guidingProbability);
                sampleBSDF = true;
            }
        }

        if (sampleBSDF){
            bs = m_bsdf->Sample_f(woRender, u, u2, mode, sampleFlags);
            if(bs && useGuiding) {
                pgl_vec3f pglwi = openpgl::cpp::Vector3(bs->wi[0], bs->wi[1], bs->wi[2]);
                float guidedPDF = m_surfaceSamplingDistribution->PDF(pglwi);
                bs->bsdfPdf = bs->pdf;
                bs->pdf = ((1.0f - guidingProbability) * bs->pdf) + (guidingProbability * guidedPDF);
                bs->misPdf = bs->pdf;
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
				bs->bsdfPdf = bsdfPDF;
                bs->misPdf = pdf;
            }
        }
        return bs;
    }

    pstd::optional<BSDFSample> Sample_f_RIS(
        Vector3f woRender, Float u, Point2f u2,
        TransportMode mode = TransportMode::Radiance,
        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {

        pstd::optional<BSDFSample> bs = {};
        bool sampleBSDF = true;
        if (!useGuiding) {
            return m_bsdf->Sample_f(woRender, u, u2, mode, sampleFlags);
        }

        const float uniformIncomingRadiancePDF = ((1.0f/(4.0f * M_PI))); 
        RISSample risSamples[2];
        // RIS0 - sample BSDF
        pstd::optional<BSDFSample> bs0 = m_bsdf->Sample_f(woRender, u, u2, mode, sampleFlags);
        if(bs0) {
            risSamples[0].f = bs0->f;
            risSamples[0].eta = bs0->eta;
            risSamples[0].sampledRoughness = bs0->sampledRoughness;
            risSamples[0].bsdfPDF = bs0->pdf;
            risSamples[0].wiRender = bs0->wi;
            pgl_vec3f pglwi0 = openpgl::cpp::Vector3(bs0->wi[0], bs0->wi[1], bs0->wi[2]);
            risSamples[0].guidingPDF = m_surfaceSamplingDistribution->PDF(pglwi0);
            risSamples[0].incomingRadiancePDF = m_surfaceSamplingDistribution->IncomingRadiancePDF(pglwi0);
            risSamples[0].flags = bs0->flags;
            risSamples[0].misPDF = 0.5f * (risSamples[0].bsdfPDF + risSamples[0].guidingPDF);
        }
        // RIS1 - sample guiding
        Point2f sample2D1 = m_sampler->Get2D();
        pgl_point2f pglSample1 = openpgl::cpp::Point2(sample2D1[0], sample2D1[1]);
        pgl_vec3f pglwi1;
        risSamples[1].guidingPDF = m_surfaceSamplingDistribution->SamplePDF(pglSample1, pglwi1);
        risSamples[1].incomingRadiancePDF = m_surfaceSamplingDistribution->IncomingRadiancePDF(pglwi1);
        Vector3f wiRender = Vector3f(pglwi1.x, pglwi1.y, pglwi1.z);
        risSamples[1].f = m_bsdf->f(woRender, wiRender, mode);
        risSamples[1].eta = m_bsdf->GetEta();
        risSamples[1].sampledRoughness = m_bsdf->GetRoughness();
        risSamples[1].bsdfPDF = m_bsdf->PDF(woRender, wiRender);
        risSamples[1].wiRender = wiRender;
        risSamples[1].flags = m_bsdf->Flags();
        risSamples[1].misPDF = 0.5f * (risSamples[1].bsdfPDF + risSamples[1].guidingPDF);

        // Calculating RIS weights
        float sumWeightsRIS = 0.f;
        int numSamplesRIS = 0;
        if(risSamples[0].bsdfPDF > 0.f) {
            risSamples[0].risWeight = (risSamples[0].bsdfPDF * ((1.0f - guidingProbability) * uniformIncomingRadiancePDF + guidingProbability * risSamples[0].incomingRadiancePDF));
            risSamples[0].risWeight /= risSamples[0].misPDF;
            sumWeightsRIS += risSamples[0].risWeight;
            numSamplesRIS++;
        }
        
        if(risSamples[1].bsdfPDF > 0.f) {
            risSamples[1].risWeight = (risSamples[1].bsdfPDF * ((1.0f - guidingProbability) * uniformIncomingRadiancePDF + guidingProbability * risSamples[1].incomingRadiancePDF));
            risSamples[1].risWeight /= risSamples[1].misPDF;
            sumWeightsRIS += risSamples[1].risWeight;
            numSamplesRIS++;
        }

        // Checking if there is any valid sample
        if(numSamplesRIS == 0 || sumWeightsRIS <=0.f)
        {
            return bs;
        }

        // Selecting RIS sample
        int idxRIS = 0;
        float sample1DRIS = sumWeightsRIS * m_sampler->Get1D();
        float sumRis = 0.f;
        for(int i = 0; i < 2; i++)
        {
            sumRis += risSamples[i].risWeight;
            if(sample1DRIS <= sumRis)
            {
                idxRIS = i;
                break;
            }
        }
        
        // calculating pseudo/stochastic PDF for the selected sample
        float pdf = (risSamples[idxRIS].risWeight * risSamples[idxRIS].misPDF) * (float(2)/ sumWeightsRIS);
        
        // PDF used for MIS with NEE (1 BSDF and 1 guiding sample)
        float misPdf = risSamples[idxRIS].misPDF;
        bs = BSDFSample(risSamples[idxRIS].f, risSamples[idxRIS].wiRender, pdf, risSamples[idxRIS].flags, risSamples[idxRIS].sampledRoughness, risSamples[idxRIS].eta, false);
        bs->bsdfPdf = risSamples[idxRIS].bsdfPDF;
        bs->misPdf = misPdf;

        return bs;
    }

    pstd::optional<BSDFSample> Sample_f(
        Vector3f woRender, Float u, Point2f u2,
        TransportMode mode = TransportMode::Radiance,
        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {

        if(guidingType == EBSDFGuideMIS) { 
            return Sample_f_MIS(woRender, u, u2, mode, sampleFlags);
        } else { // RIS
            return Sample_f_RIS(woRender, u, u2, mode, sampleFlags);
        }
    }

    float PDF(Vector3f woRender, Vector3f wiRender,
              TransportMode mode = TransportMode::Radiance,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        
        float bsdfPDF = m_bsdf->PDF(woRender, wiRender); 
        if (useGuiding){
            float pdf = 0.f;
            pgl_vec3f pglwi = openpgl::cpp::Vector3(wiRender[0], wiRender[1], wiRender[2]);
            float guidedPDF = m_surfaceSamplingDistribution->PDF(pglwi);
            if(guidingType == EBSDFGuideMIS) { 
                pdf = ((1.0f - guidingProbability) * bsdfPDF) + (guidingProbability * guidedPDF);
            } else { // RIS
                pdf = (0.5f * bsdfPDF) + (0.5f * guidedPDF);
            }
            return pdf;
        } else {
            return bsdfPDF;
        }
    }

private:
    bool m_enableGuiding = true;
    float guidingProbability = 0.5f;
    bool useGuiding = false;

    BSDFGuidingType guidingType {EBSDFGuideRIS};

    openpgl::cpp::Field* m_guiding_field;
    openpgl::cpp::SurfaceSamplingDistribution* m_surfaceSamplingDistribution;
    const BSDF* m_bsdf;
    Sampler* m_sampler;
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
