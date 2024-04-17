/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2013 Intel Corporation. */

#ifndef PBRT_GUIDING_H
#define PBRT_GUIDING_H


#include <pbrt/interaction.h>
#include <openpgl/cpp/OpenPGL.h>

#if defined(_MSC_VER)
 // Make MS math.h define M_PI
 #define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <iostream>

//#define GUIDING_SPECTRAL_TO_VEC3_USE_RGB

namespace pbrt {

inline Vector3f spectral_to_vec3(const SampledSpectrum& spec, const SampledWavelengths &lambda, const RGBColorSpace &colorSpace){

#ifndef GUIDING_SPECTRAL_TO_VEC3_USE_RGB
    const float maxSpec = std::max(std::max(std::max(spec[0], spec[1]), spec[2]), spec[3]);
    return Vector3f(maxSpec, maxSpec, maxSpec);
#else
    const RGB specRGB = spec.ToRGB(lambda, colorSpace);
    return Vector3f(specRGB[0], specRGB[1], specRGB[2]);
#endif
}

enum GuidingType{
    EGuideMIS = 0,
    EGuideRIS
};

struct GuidedBSDF{

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
    openpgl::cpp::SurfaceSamplingDistribution* surfaceSamplingDistribution, bool enableGuiding = true, GuidingType guidingType = EGuideRIS){
        m_guiding_field = guiding_field;
        m_surfaceSamplingDistribution = surfaceSamplingDistribution;
        m_enableGuiding = enableGuiding;
        m_sampler = sampler;
        m_guidingType = guidingType;
    }

    bool init(const BSDF* bsdf, const RayDifferential& ray, pstd::optional<pbrt::ShapeIntersection> si, float &rand){
        m_bsdf = bsdf;
        const Point3f p = ray.o + si->tHit * ray.d;
        pgl_point3f pglP = openpgl::cpp::Point3(p[0], p[1], p[2]);
        bool success = false;

        if (IsNonSpecular(bsdf->Flags()) ) {
            if(m_surfaceSamplingDistribution->Init(m_guiding_field, pglP, rand)){                
                // only apply the cosine product on opaque surfaces
                if(!IsTransmissive(bsdf->Flags())) {
                    Normal3f n = si->intr.shading.n;
                    // check for flipped normals
                    if(Dot(-ray.d, si->intr.shading.n) < 0.f){
                        n = -n;
                    }
                    pgl_point3f pglN = openpgl::cpp::Vector3(n[0], n[1], n[2]);
                    m_surfaceSamplingDistribution->ApplyCosineProduct(pglN);
                }
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

        if(m_guidingType == EGuideMIS) { 
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
            if(m_guidingType == EGuideMIS) { 
                pdf = ((1.0f - guidingProbability) * bsdfPDF) + (guidingProbability * guidedPDF);
            } else { // RIS
                pdf = (0.5f * bsdfPDF) + (0.5f * guidedPDF);
            }
            return pdf;
        } else {
            return bsdfPDF;
        }
    }

    uint32_t getId() const {
        return m_surfaceSamplingDistribution->GetId();
    }

private:
    bool m_enableGuiding = true;
    float guidingProbability = 0.5f;
    bool useGuiding = false;

    GuidingType m_guidingType {EGuideRIS};

    openpgl::cpp::Field* m_guiding_field;
    openpgl::cpp::SurfaceSamplingDistribution* m_surfaceSamplingDistribution;
    const BSDF* m_bsdf;
    Sampler* m_sampler;
};

struct GuidedPhaseFunction{

    struct RISSample{
        float phasePDF {0.f};
        float guidingPDF {0.f};
        float misPDF {0.f};
        float incomingRadiancePDF {0.f};
        float p;
        Vector3f wiRender;
        float meanCosine{0.0f};
        float risWeight{0.f};
    };

    GuidedPhaseFunction(Sampler* sampler, openpgl::cpp::Field* guiding_field,
    openpgl::cpp::VolumeSamplingDistribution* volumeSamplingDistribution, bool enableGuiding = true, GuidingType guidingType = EGuideMIS){
        m_guiding_field = guiding_field;
        m_volumeSamplingDistribution = volumeSamplingDistribution;
        m_enableGuiding = enableGuiding;
        m_sampler = sampler;
        m_guidingType = guidingType;
    }

    bool init(const PhaseFunction* phase, const Point3f& p, const Vector3f& wo, float &rand){
        m_phase = phase;
        pgl_point3f pglP = openpgl::cpp::Point3(p[0], p[1], p[2]);
        bool success = false;

        if(m_volumeSamplingDistribution->Init(m_guiding_field, pglP, rand)){
            pgl_vec3f pglWo = openpgl::cpp::Vector3(wo[0], wo[1], wo[2]);
            Float meanCosine = m_phase->MeanCosine();
            m_volumeSamplingDistribution->ApplySingleLobeHenyeyGreensteinProduct(pglWo,meanCosine);
            success = true;
        }

        useGuiding = m_enableGuiding ? success : false;
        return success;
    }

    Float p(Vector3f woRender, Vector3f wiRender) const {
        return m_phase->p(woRender, wiRender);
    }

    pstd::optional<PhaseFunctionSample> Sample_p_MIS(
        Vector3f woRender, Point2f u) const {

        pstd::optional<PhaseFunctionSample> ps = {};
        bool samplePhase = true;
        if (useGuiding) {
            if(guidingProbability > u.x) {
                u.x /= guidingProbability;
                samplePhase = false;
            } else {
                u.x -= guidingProbability;
                u.x /= (1.0f - guidingProbability);
                samplePhase = true;
            }
        }

        if (samplePhase){
            ps = m_phase->Sample_p(woRender, u);
            if(ps && useGuiding) {
                pgl_vec3f pglwi = openpgl::cpp::Vector3(ps->wi[0], ps->wi[1], ps->wi[2]);
                float guidedPDF = m_volumeSamplingDistribution->PDF(pglwi);
                ps->phasePdf = ps->pdf;
                ps->pdf = ((1.0f - guidingProbability) * ps->pdf) + (guidingProbability * guidedPDF);
                ps->misPdf = ps->pdf;
            }
        } else {
            pgl_point2f sample2D = openpgl::cpp::Point2(u[0], u[1]);
            pgl_vec3f pglwi;
            float guidedPDF = m_volumeSamplingDistribution->SamplePDF(sample2D, pglwi);
            
            Vector3f wiRender = Vector3f(pglwi.x, pglwi.y, pglwi.z);
            Float p = m_phase->p(woRender, wiRender);
            Float phasePDF = m_phase->PDF(woRender, wiRender);
            if(phasePDF > 0.f) {
                Float meanCosine = m_phase->MeanCosine();
                bool pdfIsProportional = false;

                float pdf = ((1.0f - guidingProbability) * phasePDF) + (guidingProbability * guidedPDF); 
                ps = PhaseFunctionSample{p, wiRender, meanCosine, pdf, phasePDF, pdf};
            }
        }
        return ps;
    }

    pstd::optional<PhaseFunctionSample> Sample_p_RIS(
        Vector3f woRender, Point2f u) const {

        pstd::optional<PhaseFunctionSample> ps = {};

        bool samplePhase = true;
        if (!useGuiding) {
            return m_phase->Sample_p(woRender, u);
        }

        const float uniformIncomingRadiancePDF = ((1.0f/(4.0f * M_PI))); 
        RISSample risSamples[2];
        // RIS0 - sample BSDF
        pstd::optional<PhaseFunctionSample> ps0 = m_phase->Sample_p(woRender, u);
        if(ps0) {
            risSamples[0].p = ps0->p;
            risSamples[0].meanCosine = ps0->meanCosine;
            risSamples[0].phasePDF = ps0->pdf;
            risSamples[0].wiRender = ps0->wi;
            pgl_vec3f pglwi0 = openpgl::cpp::Vector3(ps0->wi[0], ps0->wi[1], ps0->wi[2]);
            risSamples[0].guidingPDF = m_volumeSamplingDistribution->PDF(pglwi0);
            risSamples[0].incomingRadiancePDF = m_volumeSamplingDistribution->IncomingRadiancePDF(pglwi0);
            risSamples[0].misPDF = 0.5f * (risSamples[0].phasePDF + risSamples[0].guidingPDF);
        }
        // RIS1 - sample guiding
        Point2f sample2D1 = m_sampler->Get2D();
        pgl_point2f pglSample1 = openpgl::cpp::Point2(sample2D1[0], sample2D1[1]);
        pgl_vec3f pglwi1;
        risSamples[1].guidingPDF = m_volumeSamplingDistribution->SamplePDF(pglSample1, pglwi1);
        risSamples[1].incomingRadiancePDF = m_volumeSamplingDistribution->IncomingRadiancePDF(pglwi1);
        Vector3f wiRender = Vector3f(pglwi1.x, pglwi1.y, pglwi1.z);
        risSamples[1].p = m_phase->p(woRender, wiRender);
        risSamples[1].meanCosine = m_phase->MeanCosine();
        risSamples[1].phasePDF = m_phase->PDF(woRender, wiRender);
        risSamples[1].wiRender = wiRender;
        risSamples[1].misPDF = 0.5f * (risSamples[1].phasePDF + risSamples[1].guidingPDF);

        // Calculating RIS weights
        float sumWeightsRIS = 0.f;
        int numSamplesRIS = 0;
        if(risSamples[0].phasePDF > 0.f) {
            risSamples[0].risWeight = (risSamples[0].phasePDF * ((1.0f - guidingProbability) * uniformIncomingRadiancePDF + guidingProbability * risSamples[0].incomingRadiancePDF));
            risSamples[0].risWeight /= risSamples[0].misPDF;
            sumWeightsRIS += risSamples[0].risWeight;
            numSamplesRIS++;
        }
        
        if(risSamples[1].phasePDF > 0.f) {
            risSamples[1].risWeight = (risSamples[1].phasePDF * ((1.0f - guidingProbability) * uniformIncomingRadiancePDF + guidingProbability * risSamples[1].incomingRadiancePDF));
            risSamples[1].risWeight /= risSamples[1].misPDF;
            sumWeightsRIS += risSamples[1].risWeight;
            numSamplesRIS++;
        }

        // Checking if there is any valid sample
        if(numSamplesRIS == 0 || sumWeightsRIS <=0.f)
        {
            return ps;
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
        ps = PhaseFunctionSample{risSamples[idxRIS].p, risSamples[idxRIS].wiRender, 
                        risSamples[idxRIS].meanCosine, pdf, risSamples[idxRIS].phasePDF, misPdf};

        return ps;
    }

    pstd::optional<PhaseFunctionSample> Sample_p(
        Vector3f woRender, Point2f u) const {

        if(m_guidingType == EGuideMIS) { 
            return Sample_p_MIS(woRender, u);
        } else { // RIS
            return Sample_p_RIS(woRender, u);
        }
    }

    float PDF(Vector3f woRender, Vector3f wiRender) const {
        
        float phasePDF = m_phase->PDF(woRender, wiRender); 
        if (useGuiding){
            float pdf = 0.f;
            pgl_vec3f pglwi = openpgl::cpp::Vector3(wiRender[0], wiRender[1], wiRender[2]);
            float guidedPDF = m_volumeSamplingDistribution->PDF(pglwi);
            if(m_guidingType == EGuideMIS) { 
                pdf = ((1.0f - guidingProbability) * phasePDF) + (guidingProbability * guidedPDF);
            } else { // RIS
                pdf = (0.5f * phasePDF) + (0.5f * guidedPDF);
            }
            return pdf;
        } else {
            return phasePDF;
        }
    }

    float MeanCosine() const {
        return m_phase->MeanCosine();
    }

private:
    bool m_enableGuiding = true;
    float guidingProbability = 0.5f;
    bool useGuiding = false;

    GuidingType m_guidingType {EGuideMIS};

    openpgl::cpp::Field* m_guiding_field;
    openpgl::cpp::VolumeSamplingDistribution* m_volumeSamplingDistribution;
    const PhaseFunction* m_phase;
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

inline openpgl::cpp::PathSegment* guiding_newVolumePathSegment(openpgl::cpp::PathSegmentStorage* pathSegmentStorage, const Point3f& pos, const Vector3f& wo)
{    
    const pgl_vec3f pglZero = openpgl::cpp::Vector3(0.0f, 0.0f, 0.0f);
    const pgl_vec3f pglOne = openpgl::cpp::Vector3(1.0f, 1.0f, 1.0f);

    pgl_point3f pglP = openpgl::cpp::Point3(pos[0], pos[1], pos[2]);
    pgl_vec3f pglNormal = openpgl::cpp::Vector3(0, 0, 1);
    pgl_vec3f pglWo = openpgl::cpp::Vector3(wo[0], wo[1], wo[2]);

    openpgl::cpp::PathSegment* pathSegmentData = pathSegmentStorage->NextSegment();
    openpgl::cpp::SetPosition(pathSegmentData, pglP);
    openpgl::cpp::SetNormal(pathSegmentData, pglNormal);
    openpgl::cpp::SetDirectionOut(pathSegmentData, pglWo);
    openpgl::cpp::SetVolumeScatter(pathSegmentData, true);
    openpgl::cpp::SetScatteredContribution(pathSegmentData, pglZero);
    openpgl::cpp::SetDirectContribution(pathSegmentData, pglZero);
    openpgl::cpp::SetTransmittanceWeight(pathSegmentData, pglOne);
    openpgl::cpp::SetEta(pathSegmentData, 1.0);
    return pathSegmentData;
}

inline void guiding_addScatteredDirectLight(openpgl::cpp::PathSegment* pathSegmentData, const SampledSpectrum& Ld, SampledWavelengths &lambda, const RGBColorSpace *colorSpace)
{
    if(pathSegmentData) {
        const Vector3f LdVec3 = spectral_to_vec3(Ld, lambda, *colorSpace);
        const pgl_vec3f pglLd = openpgl::cpp::Vector3(std::max(0.f, LdVec3.x), std::max(0.f, LdVec3.y), std::max(0.f, LdVec3.z));
        openpgl::cpp::AddScatteredContribution(pathSegmentData, pglLd);
    }
}

inline void guiding_addSurfaceEmission(openpgl::cpp::PathSegment* pathSegmentData, const SampledSpectrum& Le, float misWeight, SampledWavelengths &lambda, const RGBColorSpace *colorSpace)
{
    if(pathSegmentData) {
        const Vector3f LeVec3 = spectral_to_vec3(Le, lambda, *colorSpace);
        const pgl_vec3f pglLe = openpgl::cpp::Vector3(std::max(0.f, LeVec3.x), std::max(0.f, LeVec3.y), std::max(0.f, LeVec3.z));
        openpgl::cpp::SetDirectContribution(pathSegmentData, pglLe);
        openpgl::cpp::SetMiWeight(pathSegmentData, misWeight);
    }
}

inline void guiding_addTransmittanceWeight(openpgl::cpp::PathSegment* pathSegmentData, const SampledSpectrum& transmittance, SampledWavelengths &lambda, const RGBColorSpace *colorSpace)
{
    if(pathSegmentData) {
        const Vector3f transmittanceVec3 = spectral_to_vec3(transmittance, lambda, *colorSpace);
        const pgl_vec3f pglTransmittance = openpgl::cpp::Vector3(std::max(0.f, transmittanceVec3.x), std::max(0.f, transmittanceVec3.y), std::max(0.f, transmittanceVec3.z));
        //std::cout << "transmittance: " << transmittance[0] << "\t" << transmittance[1] << "\t" << transmittance[2] << "\t" << transmittance[3] << std::endl;
        //std::cout << "pglTransmittance: " << transmittanceRGB.r << "\t" << transmittanceRGB.g << "\t" << transmittanceRGB.b << std::endl;
        openpgl::cpp::SetTransmittanceWeight(pathSegmentData, pglTransmittance);
    }
}

inline void guiding_addInfiniteLightEmission(openpgl::cpp::PathSegmentStorage* pathSegmentStorage, float guidingInfiniteLightDistance, const RayDifferential& ray, const SampledSpectrum& Le, float misWeight, SampledWavelengths &lambda, const RGBColorSpace *colorSpace)
{
    const pgl_vec3f pglZero = openpgl::cpp::Vector3(0.f, 0.f, 0.f);
    const pgl_vec3f pglOne = openpgl::cpp::Vector3(1.f, 1.f, 1.f);
    
    openpgl::cpp::PathSegment* pathSegmentData = pathSegmentStorage->NextSegment();
    if(pathSegmentData) {
        const Vector3f LeVec3 = spectral_to_vec3(Le, lambda, *colorSpace);
        const pgl_vec3f pglLe = openpgl::cpp::Vector3(std::max(0.f, LeVec3.x), std::max(0.f, LeVec3.y), std::max(0.f, LeVec3.z));

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
        const Vector3f bsdfWeightVec3 = spectral_to_vec3(bsdfWeight, lambda, *colorSpace);
        const pgl_vec3f pglBsdfWeight = openpgl::cpp::Vector3(std::max(0.f, bsdfWeightVec3.x), std::max(0.f, bsdfWeightVec3.y), std::max(0.f, bsdfWeightVec3.z));
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

inline void guiding_addVolumeData(openpgl::cpp::PathSegment* pathSegmentData, const Float& phaseWeight, const Vector3f& wi, const Float phasePDF, const Float meanCosine)
{
    const pgl_vec3f pglZero = openpgl::cpp::Vector3(0.0f, 0.0f, 0.0f);
    const pgl_vec3f pglOne = openpgl::cpp::Vector3(1.0f, 1.0f, 1.0f);
    
    if(pathSegmentData) {
        float sampledRoughness = 1.0f - std::fabs(meanCosine);
        bool is_delta = sampledRoughness < 0.001f;
        const pgl_vec3f pglPhaseWeight = openpgl::cpp::Vector3(phaseWeight, phaseWeight, phaseWeight);
        const pgl_vec3f pglWi = openpgl::cpp::Vector3(wi[0], wi[1], wi[2]);

        openpgl::cpp::SetTransmittanceWeight(pathSegmentData, pglOne);
        openpgl::cpp::SetVolumeScatter(pathSegmentData, true);
        //openpgl::cpp::SetNormal(pathSegmentData, guiding_vec3f(normal));
        openpgl::cpp::SetDirectionIn(pathSegmentData, pglWi);
        openpgl::cpp::SetPDFDirectionIn(pathSegmentData, phasePDF);
        openpgl::cpp::SetScatteringWeight(pathSegmentData, pglPhaseWeight);
        openpgl::cpp::SetIsDelta(pathSegmentData, is_delta);
        openpgl::cpp::SetEta(pathSegmentData, 1.0f);
        openpgl::cpp::SetRoughness(pathSegmentData, sampledRoughness);
        openpgl::cpp::SetRussianRouletteProbability(pathSegmentData, 1.0f);
    }
}

}  // namespace pbrt

#endif  // PBRT_GUIDING_H
