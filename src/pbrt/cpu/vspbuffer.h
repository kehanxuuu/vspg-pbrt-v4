/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2013 Intel Corporation. */

#ifndef PBRT_VSP_BUFFER_H
#define PBRT_VSP_BUFFER_H

#include <pbrt/denoiser.h>
#include <pbrt/util/image.h>
#include <pbrt/util/parallel.h>

#define USE_PVOL_EST
//#define USE_PVOL_CORRECTION

namespace pbrt {

struct VSPBuffer {

    struct Sample {
        RGB color {0.f, 0.f, 0.f};
        RGB albedo {0.f,0.f,0.f};
        Normal3f normal {0.f, 0.f, -1.f};
        bool isVolume {false};
    };

    struct Buffers {

        Buffers(Vector2i resolution): numPixels(resolution[0]*resolution[1]) {
            contribution = new RGB[numPixels];
            secondMoment = new RGB[numPixels];

            albedo = new RGB[numPixels];
            normal = new Normal3f[numPixels];
            spp = new Float[numPixels];

            filteredContribution = new RGB[numPixels];
            filteredSecondMoment = new RGB[numPixels];

            pbrt::ParallelFor(
                0, numPixels,
                PBRT_CPU_GPU_LAMBDA(int i) {
                    contribution[i] = {0.f,0.f,0.f};
                    secondMoment[i] = {0.f,0.f,0.f};

                    albedo[i] = {0.f,0.f,0.f};
                    normal[i] = {0.f,0.f,0.f};
                    spp[i] = 0.f;

                    filteredContribution[i] = {0.f,0.f,0.f};
                    filteredSecondMoment[i] = {0.f,0.f,0.f};
                });
        } 
        
        ~Buffers() {
            delete[] contribution;
            delete[] secondMoment;

            delete[] albedo;
            delete[] normal;
            delete[] spp;

            delete[] filteredContribution;
            delete[] filteredSecondMoment;
        }

        int numPixels {0};

        // sbuffer for the mean contribution
        RGB* contribution {nullptr};
        // buffer for the mean second moment (contribution^2)
        RGB* secondMoment {nullptr};

        ////// denoising feature buffers
        // buffer for the albedo (surface = material albedo, volumes = volume albedo)
        RGB* albedo {nullptr};
        // buffer for the normal (surface = material normal, volumes = inverse primary ray direction)
        Normal3f* normal {nullptr};
        // buffer for the number of samples per buffer
        Float* spp {nullptr};

        ////// filtered buffers
        // filtered buffer for the contribution
        RGB* filteredContribution {nullptr};
        // filtered buffer for the sqrt of the second moment
        RGB* filteredSecondMoment {nullptr};

    };

    VSPBuffer(const Vector2i& resolution): resolution(resolution){
        int numPixels = resolution[0]*resolution[1];
        surfaceBuffers = new Buffers(resolution);
        volumeBuffers = new Buffers(resolution);

        pVolBuffer = new Float[numPixels];
        filteredPVolBuffer = new Float[numPixels];

        vspContributionBuffer = new Float[numPixels];
        vspSecondMomentBuffer = new Float[numPixels];
        denoiser = new OIDNDenoiser(resolution, true);
        isReady = false;
    }

    VSPBuffer(const std::string& fileName) {
        Load(fileName);
        denoiser = new OIDNDenoiser(resolution, true);
        isReady = true;
    }

    ~VSPBuffer() {
        delete surfaceBuffers;
        delete volumeBuffers;

        delete[] pVolBuffer;
        delete[] filteredPVolBuffer;

        delete[] vspContributionBuffer;
        delete[] vspSecondMomentBuffer;

        delete denoiser;
    }

    /**
     * Updates the VSP buffer. 
     * First the PVol ist estimated from the volume and surface samples and then denoised.
     * Second the buffers for the contribution and sqrt second moment are denoised.
     * Third the VSP buffer for contribution and variance-aware (second moment) are calculated.
     */
    void Update(){
        std::cout << "VSPBuffer::Update()" << std::endl;
        Point2i pMin = Point2i(0,0);
        Point2i pMax = Point2i(resolution.x, resolution.y);
        Bounds2i pixelBounds = Bounds2i(pMin, pMax);

        ParallelFor2D(pixelBounds, [&](Point2i p) {
            int pIdx = p.y * resolution.x + p.x;
            const Float surfaceSampleCount = surfaceBuffers->spp[pIdx];
            const Float volumeSampleCount = volumeBuffers->spp[pIdx];
            const Float pVolEst = volumeSampleCount / (surfaceSampleCount + volumeSampleCount);
		    pVolBuffer[pIdx] = pVolEst;
        });

        denoiser->Denoise(pVolBuffer, filteredPVolBuffer);
        denoiser->Denoise(surfaceBuffers->contribution, surfaceBuffers->secondMoment, surfaceBuffers->normal, surfaceBuffers->albedo, surfaceBuffers->filteredContribution, surfaceBuffers->filteredSecondMoment);
        denoiser->Denoise(volumeBuffers->contribution, volumeBuffers->secondMoment, volumeBuffers->normal, volumeBuffers->albedo, volumeBuffers->filteredContribution, volumeBuffers->filteredSecondMoment);
        ParallelFor2D(pixelBounds, [&](Point2i p) {
            int pIdx = p.y * resolution.x + p.x;
		    Float pVolEst = filteredPVolBuffer[pIdx];
#ifndef USE_PVOL_EST
            // If we add zero samples to the other buffers (e.g., to the volume buffer when we have a surface sample)
            // Then we do not need to multiply with (1.f -pVolEst) or pVolEst.
            // Note for the second moment look for the USE_PVOL_CORRECTION earlier whne calcualting the sqrt of the second moment.
            RGB surfaceContribution = surfaceBuffers->filteredContribution[pIdx];
            RGB surfaceSecondMoment = surfaceBuffers->filteredSecondMomentSqrt[pIdx];

            RGB volumeContribution = volumeBuffers->filteredContribution[pIdx];
            RGB volumeSecondMoment = volumeBuffers->filteredSecondMomentSqrt[pIdx];
#else
            // If the surface/volume buffers only include volume or surface samples we have
            // to correct with (1.f -pVolEst) and pVolEst.
            // Not since we already use the sqrt of the second moment we only need to multiply
            // with pVolEst and not pVolEst°2 
            RGB surfaceContribution = (1.f - pVolEst) * surfaceBuffers->filteredContribution[pIdx];

            RGB surfaceSecondMomentSqrtPsurf;
            surfaceSecondMomentSqrtPsurf[0] = surfaceBuffers->filteredSecondMoment[pIdx][0] > 0.f ? (1.f - pVolEst) * std::sqrt(surfaceBuffers->filteredSecondMoment[pIdx][0]) : 0.f;
            surfaceSecondMomentSqrtPsurf[1] = surfaceBuffers->filteredSecondMoment[pIdx][1] > 0.f ? (1.f - pVolEst) * std::sqrt(surfaceBuffers->filteredSecondMoment[pIdx][1]) : 0.f;
            surfaceSecondMomentSqrtPsurf[2] = surfaceBuffers->filteredSecondMoment[pIdx][2] > 0.f ? (1.f - pVolEst) * std::sqrt(surfaceBuffers->filteredSecondMoment[pIdx][2]) : 0.f;

            RGB volumeContribution = pVolEst * volumeBuffers->filteredContribution[pIdx];

            RGB volumeSecondMomentSqrtPvol;
            volumeSecondMomentSqrtPvol[0] = volumeBuffers->filteredSecondMoment[pIdx][0] > 0.f ? pVolEst * std::sqrt(volumeBuffers->filteredSecondMoment[pIdx][0]) : 0.f;
            volumeSecondMomentSqrtPvol[1] = volumeBuffers->filteredSecondMoment[pIdx][1] > 0.f ? pVolEst * std::sqrt(volumeBuffers->filteredSecondMoment[pIdx][1]) : 0.f;
            volumeSecondMomentSqrtPvol[2] = volumeBuffers->filteredSecondMoment[pIdx][2] > 0.f ? pVolEst * std::sqrt(volumeBuffers->filteredSecondMoment[pIdx][2]) : 0.f;

#endif
            RGB contribution = surfaceContribution + volumeContribution;
            RGB secondMoment = surfaceSecondMomentSqrtPsurf + volumeSecondMomentSqrtPvol;

            Float contributionScalar = contribution.MaxValue();
            Float secondMomentScalar = secondMoment.MaxValue();

            Float volumeContributionScalar = volumeContribution.MaxValue();
            Float volumeSecondMomentScalar = volumeSecondMomentSqrtPvol.MaxValue();

		    vspContributionBuffer[pIdx] = contributionScalar > 0.f ? volumeContributionScalar / (contributionScalar) : -1.f;
		    vspSecondMomentBuffer[pIdx] = secondMomentScalar > 0.f ? volumeSecondMomentScalar / (secondMomentScalar) : -1.f;
        });

        isReady = true;
    }

    /**
     * Add a sample to either the surface or the volume buffer
     */
    void AddSample(Point2i pPixel, const Sample& sample) {
        int pixIdx = pPixel.y * resolution.x + pPixel.x;
        RGB color = {std::min(sample.color[0], maxColor), std::min(sample.color[1], maxColor), std::min(sample.color[2], maxColor)};
        if(!sample.isVolume) {
            surfaceBuffers->spp[pixIdx] += 1;
#ifdef USE_PVOL_EST
            // calculating the alpha using only the number of surface samples
            float alpha = 1.f / surfaceBuffers->spp[pixIdx];
#else
            // calculating the alpha simulating we added zero samples for each volume sample as well 
            float alpha = 1.f / (surfaceBuffers->spp[pixIdx] + volumeBuffers->spp[pixIdx]);
#endif
            surfaceBuffers->contribution[pixIdx] = (1.f - alpha) * surfaceBuffers->contribution[pixIdx] + alpha * color;
            surfaceBuffers->albedo[pixIdx] = (1.f - alpha) * surfaceBuffers->albedo[pixIdx] + alpha * sample.albedo;
            surfaceBuffers->normal[pixIdx] = (1.f - alpha) * surfaceBuffers->normal[pixIdx] + alpha * sample.normal;
            surfaceBuffers->secondMoment[pixIdx] = (1.f - alpha) * surfaceBuffers->secondMoment[pixIdx] + alpha * (color * color);
#ifndef USE_PVOL_EST
            // adding zero value samples to the volume buffer
            volumeBuffers->contribution[pixIdx] = (1.f - alpha) * volumeBuffers->contribution[pixIdx];
            volumeBuffers->secondMoment[pixIdx] = (1.f - alpha) * volumeBuffers->secondMoment[pixIdx];
#endif
        } else {
            volumeBuffers->spp[pixIdx] += 1;
#ifdef USE_PVOL_EST
            // calculating the alpha using only the number of volume samples
            float alpha = 1.f / volumeBuffers->spp[pixIdx];
#else
            // calculating the alpha simulating we added zero samples for each surface sample as well 
            float alpha = 1.f / (surfaceBuffers->spp[pixIdx] + volumeBuffers->spp[pixIdx]);
#endif
            volumeBuffers->contribution[pixIdx] = (1.f - alpha) * volumeBuffers->contribution[pixIdx] + alpha * color;
            volumeBuffers->albedo[pixIdx] = (1.f - alpha) * volumeBuffers->albedo[pixIdx] + alpha * sample.albedo;
            volumeBuffers->normal[pixIdx] = (1.f - alpha) * volumeBuffers->normal[pixIdx] + alpha * sample.normal;
            volumeBuffers->secondMoment[pixIdx] = (1.f - alpha) * volumeBuffers->secondMoment[pixIdx] + alpha * (color * color);
#ifndef USE_PVOL_EST
            // adding zero value samples to the surface buffer
            surfaceBuffers->contribution[pixIdx] = (1.f - alpha) * surfaceBuffers->contribution[pixIdx];
            surfaceBuffers->secondMoment[pixIdx] = (1.f - alpha) * surfaceBuffers->secondMoment[pixIdx];
#endif
        }
    }

    /**
     * Returns the VSP for a given pixel index.
     */
    Float GetVSP(const Point2i &pPixel, const bool contributionBased = true) const {
        const int pixIdx = pPixel.y * resolution.x + pPixel.x;
        const Float vsp = contributionBased ? vspContributionBuffer[pixIdx] : vspSecondMomentBuffer[pixIdx];
        //std::cout << "GetVSP("<< secondMoment <<"): " << "pixIdx = " << pixIdx << "    vsp = " << vsp << std::endl;
        return vsp;
    }

    bool Ready() const {
        return isReady;
    }

    void Store(const std::string& fileName) const {
        std::cout << "VSPBuffer::Store(): " << fileName << std::endl;
        PixelFormat format = PixelFormat::Float;
        Point2i pMin = Point2i(0,0);
        Point2i pMax = Point2i(resolution.x, resolution.y);
        Bounds2i pixelBounds = Bounds2i(pMin, pMax);
        Image image(format, Point2i(resolution),
                {
                 "SurfContrib.R",
                 "SurfContrib.G",
                 "SurfContrib.B",
                 "SurfContribDenoised.R",
                 "SurfContribDenoised.G",
                 "SurfContribDenoised.B",
                 "SurfSecMom.R",
                 "SurfSecMom.G",
                 "SurfSecMom.B",
                 "SurfSecMomDenoised.R",
                 "SurfSecMomDenoised.G",
                 "SurfSecMomDenoised.B",
                 "surfN.x",
                 "surfN.y",
                 "surfN.z",
                 "surfAlbedo.R",
                 "surfAlbedo.G",
                 "surfAlbedo.B",
                 "surfSPP",

                 "VolContrib.R",
                 "VolContrib.G",
                 "VolContrib.B",
                 "VolContribDenoised.R",
                 "VolContribDenoised.G",
                 "VolContribDenoised.B",
                 "VolSecMom.R",
                 "VolSecMom.G",
                 "VolSecMom.B",
                 "VolSecMomDenoised.R",
                 "VolSecMomDenoised.G",
                 "VolSecMomDenoised.B",
                 "volN.x",
                 "volN.y",
                 "volN.z",
                 "volAlbedo.R",
                 "volAlbedo.G",
                 "volAlbedo.B",
                 "volSPP",

                 "pVolActual",
                 "pVolActualDenoised",
                 "pVolTargetContrib",
                 "pVolTargetSecMom",});

        ImageChannelDesc surfContributionDesc = image.GetChannelDesc({"SurfContrib.R", "SurfContrib.G", "SurfContrib.B"});
        ImageChannelDesc surfContributionDenoisedDesc = image.GetChannelDesc({"SurfContribDenoised.R", "SurfContribDenoised.G", "SurfContribDenoised.B"});

        ImageChannelDesc surfSecondMomentDesc = image.GetChannelDesc({"SurfSecMom.R", "SurfSecMom.G", "SurfSecMom.B"});
        ImageChannelDesc surfSecondMomentDenoisedDesc = image.GetChannelDesc({"SurfSecMomDenoised.R", "SurfSecMomDenoised.G", "SurfSecMomDenoised.B"});

        ImageChannelDesc surfNormalDesc = image.GetChannelDesc({"surfN.x", "surfN.y", "surfN.z"});
        ImageChannelDesc surfAlbedoDesc = image.GetChannelDesc({"surfAlbedo.R", "surfAlbedo.G", "surfAlbedo.B"});
        ImageChannelDesc surfSPPDesc = image.GetChannelDesc({"surfSPP"});

        ImageChannelDesc volContributionDesc = image.GetChannelDesc({"VolContrib.R", "VolContrib.G", "VolContrib.B"});
        ImageChannelDesc volContributionDenoisedDesc = image.GetChannelDesc({"VolContribDenoised.R", "VolContribDenoised.G", "VolContribDenoised.B"});

        ImageChannelDesc volSecondMomentDesc = image.GetChannelDesc({"VolSecMom.R", "VolSecMom.G", "VolSecMom.B"});
        ImageChannelDesc volSecondMomentDenoisedDesc = image.GetChannelDesc({"VolSecMomDenoised.R", "VolSecMomDenoised.G", "VolSecMomDenoised.B"});

        ImageChannelDesc volNormalDesc = image.GetChannelDesc({"volN.x", "volN.y", "volN.z"});
        ImageChannelDesc volAlbedoDesc = image.GetChannelDesc({"volAlbedo.R", "volAlbedo.G", "volAlbedo.B"});
        ImageChannelDesc volSPPDesc = image.GetChannelDesc({"volSPP"});

        ImageChannelDesc pVolActualDesc = image.GetChannelDesc({"pVolActual"});
        ImageChannelDesc pVolActualDenoisedDesc = image.GetChannelDesc({"pVolActualDenoised"});
        ImageChannelDesc pVolTargetContribDesc = image.GetChannelDesc({"pVolTargetContrib"});
        ImageChannelDesc pVolTargetSecMomDesc = image.GetChannelDesc({"pVolTargetSecMom"});

        ParallelFor2D(pixelBounds, [&](Point2i p) {
            int pIdx = p.y * resolution.x + p.x;

            Float pVol = pVolBuffer[pIdx];
            Float pVolEst = filteredPVolBuffer[pIdx];
            Float pSurfEst = 1.f - pVolEst;

            RGB surfContributionPsurf = pSurfEst * surfaceBuffers->contribution[pIdx];
            RGB surfContributionPsurfDenoised = pSurfEst * surfaceBuffers->filteredContribution[pIdx];

            RGB surfSecondMomentSqrtPsurf;
            surfSecondMomentSqrtPsurf[0] = surfaceBuffers->filteredSecondMoment[pIdx][0] > 0.f ? pSurfEst * std::sqrt(surfaceBuffers->secondMoment[pIdx][0]) : 0.f;
            surfSecondMomentSqrtPsurf[1] = surfaceBuffers->filteredSecondMoment[pIdx][1] > 0.f ? pSurfEst * std::sqrt(surfaceBuffers->secondMoment[pIdx][1]) : 0.f;
            surfSecondMomentSqrtPsurf[2] = surfaceBuffers->filteredSecondMoment[pIdx][2] > 0.f ? pSurfEst * std::sqrt(surfaceBuffers->secondMoment[pIdx][2]) : 0.f;
            RGB surfSecondMomentSqrtPsurfDenoised;
            surfSecondMomentSqrtPsurfDenoised[0] = surfaceBuffers->filteredSecondMoment[pIdx][0] > 0.f ? pSurfEst * std::sqrt(surfaceBuffers->filteredSecondMoment[pIdx][0]) : 0.f;
            surfSecondMomentSqrtPsurfDenoised[1] = surfaceBuffers->filteredSecondMoment[pIdx][1] > 0.f ? pSurfEst * std::sqrt(surfaceBuffers->filteredSecondMoment[pIdx][1]) : 0.f;
            surfSecondMomentSqrtPsurfDenoised[2] = surfaceBuffers->filteredSecondMoment[pIdx][2] > 0.f ? pSurfEst * std::sqrt(surfaceBuffers->filteredSecondMoment[pIdx][2]) : 0.f;

            RGB surfNormal = (RGB(surfaceBuffers->normal[pIdx].x, surfaceBuffers->normal[pIdx].y, surfaceBuffers->normal[pIdx].z) + RGB(1.f, 1.f, 1.f)) * 0.5f;
            RGB surfAlbedo = surfaceBuffers->albedo[pIdx];
            Float surfSPP = surfaceBuffers->spp[pIdx];

            RGB volContributionPvol = pVolEst * volumeBuffers->contribution[pIdx];
            RGB volContributionPvolDenoised = pVolEst * volumeBuffers->filteredContribution[pIdx];

            RGB volSecondMomentSqrtPvol;
            volSecondMomentSqrtPvol[0] = volumeBuffers->filteredSecondMoment[pIdx][0] > 0.f ? pVolEst * std::sqrt(volumeBuffers->secondMoment[pIdx][0]) : 0.f;
            volSecondMomentSqrtPvol[1] = volumeBuffers->filteredSecondMoment[pIdx][1] > 0.f ? pVolEst * std::sqrt(volumeBuffers->secondMoment[pIdx][1]) : 0.f;
            volSecondMomentSqrtPvol[2] = volumeBuffers->filteredSecondMoment[pIdx][2] > 0.f ? pVolEst * std::sqrt(volumeBuffers->secondMoment[pIdx][2]) : 0.f;
            RGB volSecondMomentSqrtPvolDenoised;
            volSecondMomentSqrtPvolDenoised[0] = volumeBuffers->filteredSecondMoment[pIdx][0] > 0.f ? pVolEst * std::sqrt(volumeBuffers->filteredSecondMoment[pIdx][0]) : 0.f;
            volSecondMomentSqrtPvolDenoised[1] = volumeBuffers->filteredSecondMoment[pIdx][1] > 0.f ? pVolEst * std::sqrt(volumeBuffers->filteredSecondMoment[pIdx][1]) : 0.f;
            volSecondMomentSqrtPvolDenoised[2] = volumeBuffers->filteredSecondMoment[pIdx][2] > 0.f ? pVolEst * std::sqrt(volumeBuffers->filteredSecondMoment[pIdx][2]) : 0.f;

            RGB volNormal = (RGB(volumeBuffers->normal[pIdx].x, volumeBuffers->normal[pIdx].y, volumeBuffers->normal[pIdx].z) + RGB(1.f, 1.f, 1.f)) * 0.5f;
            RGB volAlbedo = volumeBuffers->albedo[pIdx];
            Float volSPP = volumeBuffers->spp[pIdx];

            Float vspContribution = vspContributionBuffer[pIdx];
            Float vspSecondMoment = vspSecondMomentBuffer[pIdx];

            Point2i pOffset(p.x, p.y);
            image.SetChannels(pOffset, surfContributionDesc, {surfContributionPsurf[0], surfContributionPsurf[1], surfContributionPsurf[2]});
            image.SetChannels(pOffset, surfContributionDenoisedDesc, {surfContributionPsurfDenoised[0], surfContributionPsurfDenoised[1], surfContributionPsurfDenoised[2]});

            image.SetChannels(pOffset, surfSecondMomentDesc, {surfSecondMomentSqrtPsurf[0], surfSecondMomentSqrtPsurf[1], surfSecondMomentSqrtPsurf[2]});
            image.SetChannels(pOffset, surfSecondMomentDenoisedDesc, {surfSecondMomentSqrtPsurfDenoised[0], surfSecondMomentSqrtPsurfDenoised[1], surfSecondMomentSqrtPsurfDenoised[2]});

            image.SetChannels(pOffset, surfNormalDesc, {surfNormal[0], surfNormal[1], surfNormal[2]});
            image.SetChannels(pOffset, surfAlbedoDesc, {surfAlbedo[0], surfAlbedo[1], surfAlbedo[2]});
            image.SetChannels(pOffset, surfSPPDesc, {surfSPP});

            image.SetChannels(pOffset, volContributionDesc, {volContributionPvol[0], volContributionPvol[1], volContributionPvol[2]});
            image.SetChannels(pOffset, volContributionDenoisedDesc, {volContributionPvolDenoised[0], volContributionPvolDenoised[1], volContributionPvolDenoised[2]});

            image.SetChannels(pOffset, volSecondMomentDesc, {volSecondMomentSqrtPvol[0], volSecondMomentSqrtPvol[1], volSecondMomentSqrtPvol[2]});
            image.SetChannels(pOffset, volSecondMomentDenoisedDesc, {volSecondMomentSqrtPvolDenoised[0], volSecondMomentSqrtPvolDenoised[1], volSecondMomentSqrtPvolDenoised[2]});

            image.SetChannels(pOffset, volNormalDesc, {volNormal[0], volNormal[1], volNormal[2]});
            image.SetChannels(pOffset, volAlbedoDesc, {volAlbedo[0], volAlbedo[1], volAlbedo[2]});
            image.SetChannels(pOffset, volSPPDesc, {volSPP});

            image.SetChannels(pOffset, pVolActualDesc, {pVol});
            image.SetChannels(pOffset, pVolActualDenoisedDesc, {pVolEst});
            image.SetChannels(pOffset, pVolTargetContribDesc, {vspContribution});
            image.SetChannels(pOffset, pVolTargetSecMomDesc, {vspSecondMoment});
        });
        image.Write(fileName);
    }
private:
    void Load(const std::string& fileName) {
        std::cout << "VSPBuffer::Load(): " << fileName << std::endl;
        ImageAndMetadata imgAndMeta = Image::Read(fileName);

        resolution = Vector2i(imgAndMeta.image.Resolution());

        int numPixels = resolution[0] * resolution[1];
        surfaceBuffers = new Buffers(resolution);
        volumeBuffers = new Buffers(resolution);

        pVolBuffer = new Float[numPixels];
        filteredPVolBuffer = new Float[numPixels];

        vspContributionBuffer = new Float[numPixels];
        vspSecondMomentBuffer = new Float[numPixels];

        ImageChannelDesc surfContributionDesc = imgAndMeta.image.GetChannelDesc({"SurfContrib.R", "SurfContrib.G", "SurfContrib.B"});
        ImageChannelDesc surfContributionDenoisedDesc = imgAndMeta.image.GetChannelDesc({"SurfContribDenoised.R", "SurfContribDenoised.G", "SurfContribDenoised.B"});
        ImageChannelDesc surfSecondMomentDesc = imgAndMeta.image.GetChannelDesc({"SurfSecMom.R", "SurfSecMom.G", "SurfSecMom.B"});
        ImageChannelDesc surfSecondMomentDenoisedDesc = imgAndMeta.image.GetChannelDesc({"SurfSecMomDenoised.R", "SurfSecMomDenoised.G", "SurfSecMomDenoised.B"});

        ImageChannelDesc surfNormalDesc = imgAndMeta.image.GetChannelDesc({"surfN.x", "surfN.y", "surfN.z"});
        ImageChannelDesc surfAlbedoDesc = imgAndMeta.image.GetChannelDesc({"surfAlbedo.R", "surfAlbedo.G", "surfAlbedo.B"});
        ImageChannelDesc surfSPPDesc = imgAndMeta.image.GetChannelDesc({"surfSPP"});

        ImageChannelDesc volContributionDesc = imgAndMeta.image.GetChannelDesc({"VolContrib.R", "VolContrib.G", "VolContrib.B"});
        ImageChannelDesc volContributionDenoisedDesc = imgAndMeta.image.GetChannelDesc({"VolContribDenoised.R", "VolContribDenoised.G", "VolContribDenoised.B"});

        ImageChannelDesc volSecondMomentDesc = imgAndMeta.image.GetChannelDesc({"VolSecMom.R", "VolSecMom.G", "VolSecMom.B"});
        ImageChannelDesc volSecondMomentDenoisedDesc = imgAndMeta.image.GetChannelDesc({"VolSecMomDenoised.R", "VolSecMomDenoised.G", "VolSecMomDenoised.B"});

        ImageChannelDesc volNormalDesc = imgAndMeta.image.GetChannelDesc({"volN.x", "volN.y", "volN.z"});
        ImageChannelDesc volAlbedoDesc = imgAndMeta.image.GetChannelDesc({"volAlbedo.R", "volAlbedo.G", "volAlbedo.B"});
        ImageChannelDesc volSPPDesc = imgAndMeta.image.GetChannelDesc({"volSPP"});

        ImageChannelDesc pVolActualDesc = imgAndMeta.image.GetChannelDesc({"pVolActual"});
        ImageChannelDesc pVolActualDenoisedDesc = imgAndMeta.image.GetChannelDesc({"pVolActualDenoised"});
        ImageChannelDesc pVolTargetContribDesc = imgAndMeta.image.GetChannelDesc({"pVolTargetContrib"});
        ImageChannelDesc pVolTargetSecMomDesc = imgAndMeta.image.GetChannelDesc({"pVolTargetSecMom"});

        Point2i pMin = Point2i(0,0);
        Point2i pMax = Point2i(resolution.x, resolution.y);
        Bounds2i pixelBounds = Bounds2i(pMin, pMax);

        ParallelFor2D(pixelBounds, [&](Point2i p) {
            int pIdx = p.y * resolution.x + p.x;

            Point2i pOffset(p.x, p.y);
            ImageChannelValues surfContributionPsurf = imgAndMeta.image.GetChannels(pOffset, surfContributionDesc);
            ImageChannelValues surfContributionPsurfDenoised = imgAndMeta.image.GetChannels(pOffset, surfContributionDenoisedDesc);

            ImageChannelValues surfSecondMomentSqrtPsurf = imgAndMeta.image.GetChannels(pOffset, surfSecondMomentDesc);
            ImageChannelValues surfSecondMomentSqrtPsurfDenoised = imgAndMeta.image.GetChannels(pOffset, surfSecondMomentDenoisedDesc);

            ImageChannelValues surfNormal = imgAndMeta.image.GetChannels(pOffset, surfNormalDesc);
            ImageChannelValues surfAlbedo = imgAndMeta.image.GetChannels(pOffset, surfAlbedoDesc);
            ImageChannelValues surfSPP = imgAndMeta.image.GetChannels(pOffset, surfSPPDesc);

            ImageChannelValues volContributionPvol = imgAndMeta.image.GetChannels(pOffset, volContributionDesc);
            ImageChannelValues volContributionPvolDenoised = imgAndMeta.image.GetChannels(pOffset, volContributionDenoisedDesc);

            ImageChannelValues volSecondMomentSqrtPvol = imgAndMeta.image.GetChannels(pOffset, volSecondMomentDesc);
            ImageChannelValues volSecondMomentSqrtPvolDenoised = imgAndMeta.image.GetChannels(pOffset, volSecondMomentDenoisedDesc);

            ImageChannelValues volNormal = imgAndMeta.image.GetChannels(pOffset, volNormalDesc);
            ImageChannelValues volAlbedo = imgAndMeta.image.GetChannels(pOffset, volAlbedoDesc);
            ImageChannelValues volSPP = imgAndMeta.image.GetChannels(pOffset, volSPPDesc);

            ImageChannelValues pVol = imgAndMeta.image.GetChannels(pOffset, pVolActualDesc);
            ImageChannelValues pVolEst = imgAndMeta.image.GetChannels(pOffset, pVolActualDenoisedDesc);
            ImageChannelValues vspContribution = imgAndMeta.image.GetChannels(pOffset, pVolTargetContribDesc);
            ImageChannelValues vspSecondMoment = imgAndMeta.image.GetChannels(pOffset, pVolTargetSecMomDesc);

            Float pVolume =  pVolEst[0];
            Float pSurface = 1 - pVolume;

            surfaceBuffers->contribution[pIdx] = {surfContributionPsurf[0] / pSurface,
                                                  surfContributionPsurf[1] / pSurface,
                                                  surfContributionPsurf[2] / pSurface};
            surfaceBuffers->filteredContribution[pIdx]= {surfContributionPsurfDenoised[0] / pSurface,
                                                         surfContributionPsurfDenoised[1] / pSurface,
                                                         surfContributionPsurfDenoised[2] / pSurface};

            surfaceBuffers->secondMoment[pIdx] = {std::pow(surfSecondMomentSqrtPsurf[0] / pSurface, 2.f),
                                                      std::pow(surfSecondMomentSqrtPsurf[1] / pSurface, 2.f),
                                                      std::pow(surfSecondMomentSqrtPsurf[2] / pSurface, 2.f)};
            surfaceBuffers->filteredSecondMoment[pIdx] = {std::pow(surfSecondMomentSqrtPsurfDenoised[0] / pSurface, 2.f),
                                                          std::pow(surfSecondMomentSqrtPsurfDenoised[1] / pSurface, 2.f),
                                                          std::pow(surfSecondMomentSqrtPsurfDenoised[2] / pSurface, 2.f)};

            surfaceBuffers->normal[pIdx] = {surfNormal[0]*2.f - 1.0f, surfNormal[1]*2.f - 1.0f, surfNormal[2]*2.f - 1.0f};
            surfaceBuffers->albedo[pIdx] = {surfAlbedo[0], surfAlbedo[1], surfAlbedo[2]};
            surfaceBuffers->spp[pIdx] = surfSPP[0];

            volumeBuffers->contribution[pIdx] = {volContributionPvol[0] / pVolume,
                                                 volContributionPvol[1] / pVolume,
                                                 volContributionPvol[2] / pVolume};
            volumeBuffers->filteredContribution[pIdx] = {volContributionPvolDenoised[0] / pVolume,
                                                         volContributionPvolDenoised[1] / pVolume,
                                                         volContributionPvolDenoised[2] / pVolume};

            volumeBuffers->secondMoment[pIdx] = {std::pow(volSecondMomentSqrtPvol[0] / pVolume, 2.f),
                                                 std::pow(volSecondMomentSqrtPvol[1] / pVolume, 2.f),
                                                 std::pow(volSecondMomentSqrtPvol[2] / pVolume, 2.f)};
            volumeBuffers->filteredSecondMoment[pIdx] = {std::pow(volSecondMomentSqrtPvolDenoised[0] / pVolume, 2.f),
                                                         std::pow(volSecondMomentSqrtPvolDenoised[1] / pVolume, 2.f),
                                                         std::pow(volSecondMomentSqrtPvolDenoised[2] / pVolume, 2.f)};

            volumeBuffers->normal[pIdx]= {volNormal[0]*2.f - 1.0f, volNormal[1]*2.f - 1.0f, volNormal[2]*2.f - 1.0f};
            volumeBuffers->albedo[pIdx] = {volAlbedo[0], volAlbedo[1], volAlbedo[2]};
            volumeBuffers->spp[pIdx] = volSPP[0];

            pVolBuffer[pIdx] = pVol[0];
            filteredPVolBuffer[pIdx] = pVolEst[0];
            vspContributionBuffer[pIdx] = vspContribution[0];
            vspSecondMomentBuffer[pIdx] = vspSecondMoment[0];
        });
    }

private:
    Vector2i resolution;

    Buffers *surfaceBuffers {nullptr};
    Buffers *volumeBuffers {nullptr};

    Float *pVolBuffer {nullptr};
    Float *filteredPVolBuffer {nullptr};

    Float *vspContributionBuffer {nullptr};
    Float *vspSecondMomentBuffer {nullptr};

    OIDNDenoiser* denoiser;
    Float maxColor {100.f};
    bool isReady {false};
};

struct TrBuffer {
    TrBuffer(const Vector2i& resolution): resolution(resolution){
        int numPixels = resolution[0]*resolution[1];
        spp = new int[numPixels];
        transmittanceBuffer = new RGB[numPixels];
    }

    TrBuffer(const std::string& fileName) {
        Load(fileName);
    }

    ~TrBuffer() {
        delete[] spp;
        delete[] transmittanceBuffer;
    }

    void AddSample(Point2i pPixel, const RGB transmittance) {
        int pixIdx = pPixel.y * resolution.x + pPixel.x;
        spp[pixIdx] += 1;
        float alpha = 1.f / spp[pixIdx];
        transmittanceBuffer[pixIdx] = (1.f - alpha) * transmittanceBuffer[pixIdx] + alpha * transmittance;
    }

    RGB GetTransmittance(const Point2i &pPixel) const {
        const int pixIdx = pPixel.y * resolution.x + pPixel.x;
        return transmittanceBuffer[pixIdx];
    }

    void Store(const std::string& fileName) const {
        std::cout << "TrBuffer::Store(): " << fileName << std::endl;
        PixelFormat format = PixelFormat::Float;
        Point2i pMin = Point2i(0,0);
        Point2i pMax = Point2i(resolution.x, resolution.y);
        Bounds2i pixelBounds = Bounds2i(pMin, pMax);
        Image image(format, Point2i(resolution),
                    {
                    "Transmittance.R",
                    "Transmittance.G",
                    "Transmittance.B",});
        ImageChannelDesc transmittanceDesc = image.GetChannelDesc({"Transmittance.R", "Transmittance.G", "Transmittance.B"});

        ParallelFor2D(pixelBounds, [&](Point2i p) {
            int pIdx = p.y * resolution.x + p.x;
            Point2i pOffset(p.x, p.y);
            image.SetChannels(pOffset, transmittanceDesc, {transmittanceBuffer[pIdx][0], transmittanceBuffer[pIdx][1], transmittanceBuffer[pIdx][2]});
        });
        image.Write(fileName);
    }
private:
    void Load(const std::string& fileNameTr) {
        std::cout << "TrBuffer::Load(): " << fileNameTr << std::endl;
        ImageAndMetadata imgAndMeta = Image::Read(fileNameTr);

        resolution = Vector2i(imgAndMeta.image.Resolution());
        int numPixels = resolution[0] * resolution[1];

        transmittanceBuffer = new RGB[numPixels];

        ImageChannelDesc transmittanceDesc = imgAndMeta.image.GetChannelDesc({"Transmittance.R", "Transmittance.G", "Transmittance.B"});

        Point2i pMin = Point2i(0,0);
        Point2i pMax = Point2i(resolution.x, resolution.y);
        Bounds2i pixelBounds = Bounds2i(pMin, pMax);

        ParallelFor2D(pixelBounds, [&](Point2i p) {
            int pIdx = p.y * resolution.x + p.x;

            Point2i pOffset(p.x, p.y);

            ImageChannelValues transmittance = imgAndMeta.image.GetChannels(pOffset, transmittanceDesc);
            transmittanceBuffer[pIdx] = {transmittance[0], transmittance[1], transmittance[2]};
        });
    }
private:
    Vector2i resolution;
    int *spp {nullptr};
    RGB *transmittanceBuffer {nullptr};
};

}

#endif
