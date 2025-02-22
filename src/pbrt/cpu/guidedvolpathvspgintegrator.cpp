#include <pbrt/cpu/integrators.h>

#include <pbrt/bsdf.h>
#include <pbrt/bssrdf.h>
#include <pbrt/cameras.h>
#include <pbrt/film.h>
#include <pbrt/filters.h>
#include <pbrt/interaction.h>
#include <pbrt/lights.h>
#include <pbrt/materials.h>
#include <pbrt/media.h>
#include <pbrt/media_sampleTMaj.h>
#include <pbrt/options.h>
#include <pbrt/paramdict.h>
#include <pbrt/samplers.h>
#include <pbrt/shapes.h>
#include <pbrt/util/bluenoise.h>
#include <pbrt/util/check.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/display.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/image.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/progressreporter.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/string.h>

#include <algorithm>

#include <iostream>

//#define VOLUME_ABSORB

namespace pbrt {

STAT_PERCENT("Integrator/Regularized BSDFs", regularizedBSDFs, totalBSDFs);

STAT_COUNTER("Integrator/Volume interactions", volumeInteractions);
STAT_COUNTER("Integrator/Surface interactions", surfaceInteractions);

STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);
STAT_INT_DISTRIBUTION("Integrator/Density query", densityQueryCount);

STAT_TIME_COUNTER("Guiding Cache Training", guidingCacheUpdateTime);
STAT_TIME_COUNTER("Image Space Guiding Buffer Training", isgBufferUpdateTime);

#ifdef PBRT_WITH_PATH_GUIDING
// GuidedVolPathVSPGIntegrator Method Definitions
GuidedVolPathVSPGIntegrator::GuidedVolPathVSPGIntegrator(int maxDepth, int minRRDepth, bool useNEE, const GuidingSettings guideSettings, const RGBColorSpace *colorSpace, Camera camera, Sampler sampler, Primitive aggregate,
                                                 std::vector<Light> lights,
                                                 const std::string &lightSampleStrategy,
                                                 bool regularize)
        : RayIntegrator(camera, sampler, aggregate, lights),
          maxDepth(maxDepth),
          minRRDepth(minRRDepth),
          useNEE(useNEE),
          guideSettings(guideSettings),
          colorSpace(colorSpace),
          lightSampler(LightSampler::Create(lightSampleStrategy, lights, Allocator())),
          regularize(regularize) {
    std::cout<< "GuidedVolPathVSPGIntegrator:" <<std::endl;
    std::cout<< "\t maxDepth = " << maxDepth << std::endl;
    std::cout<< "\t minRRDepth = " << minRRDepth << std::endl;
    std::cout<< "\t useNEE = " << useNEE << std::endl;

    std::cout<< "\t surfaceGuiding = " << guideSettings.guideSurface << std::endl;
    std::cout<< "\t volumeGuiding = " << guideSettings.guideVolume << std::endl;
    std::cout<< "\t surfaceGuidingType = " << guideSettings.surfaceGuidingType << std::endl;
    std::cout<< "\t volumeGuidingType = " << guideSettings.volumeGuidingType << std::endl;
    std::cout<< "\t storeGuidingCache = " << guideSettings.storeGuidingCache << std::endl;
    std::cout<< "\t loadGuidingCache = " << guideSettings.loadGuidingCache << std::endl;
    std::cout<< "\t guidingCacheFileName = " << guideSettings.guidingCacheFileName << std::endl << std::endl;

    std::cout<< "\t vspguiding = " << guideSettings.guideVSP << std::endl;
    std::cout<< "\t vspprimaryguiding = " << guideSettings.guidePrimaryVSP << std::endl;
    std::cout<< "\t vspsecondaryguiding = " << guideSettings.guideSecondaryVSP << std::endl;
    std::cout<< "\t vspmisratio = " << guideSettings.vspMISRatio << std::endl;
    std::cout<< "\t vspcriterion = " << (guideSettings.vspCriterion == EContribution ? "Contribution" : "Variance") << std::endl;
    std::cout<< "\t vspsamplingmethod = " << (guideSettings.guideVSPSamplingMethod == EResampling ? "Resampling" : "Villemin") << std::endl;
    std::cout<< "\t collisionProbabilityBias = " << guideSettings.collisionProbabilityBias << std::endl << std::endl;

    std::cout<< "\t storeISGBuffer = " << guideSettings.storeISGBuffer << std::endl;
    std::cout<< "\t loadISGBuffer = " << guideSettings.loadISGBuffer << std::endl;
    std::cout<< "\t isgBufferFileName = " << guideSettings.isgBufferFileName << std::endl << std::endl;

    std::cout<< "\t storeTrBuffer = " << guideSettings.storeTrBuffer << std::endl;
    std::cout<< "\t loadTrBuffer = " << guideSettings.loadTrBuffer << std::endl;
    std::cout<< "\t trBufferFileName = " << guideSettings.trBufferFileName << std::endl << std::endl;

    std::cout<< "\t rrguiding = " << guideSettings.guideRR << std::endl;
    std::cout<< "\t surfacerrguiding = " << guideSettings.guideSurfaceRR << std::endl;
    std::cout<< "\t volumerrguiding = " << guideSettings.guideVolumeRR << std::endl;

    std::cout<< "\t lightSampleStrategy = " << lightSampleStrategy << std::endl;
    std::cout<< "\t regularize = " << regularize << std::endl;

    guideTraining = guideSettings.guideSurface || guideSettings.guideVolume || guideSettings.guideSurfaceRR || guideSettings.guideVolumeRR || guideSettings.guideSecondaryVSP;

    guiding_device = new openpgl::cpp::Device(PGL_DEVICE_TYPE_CPU_4);
    guiding_fieldConfig.Init(PGL_SPATIAL_STRUCTURE_KDTREE, PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM);
#if defined(OPENPGL_VSP_GUIDING)
    guiding_fieldConfig.SetVarianceBasedVSP(guideSettings.vspCriterion == EVariance);
#endif

    if (guideSettings.loadGuidingCache) {
        if (FileExists(guideSettings.guidingCacheFileName)) {
            std::cout<< "GuidedVolPathVSPGIntegrator: loading guiding cache = "<< guideSettings.guidingCacheFileName <<std::endl;
            guiding_field = new openpgl::cpp::Field(guiding_device, guideSettings.guidingCacheFileName);
            guideTraining = false;
            enableGuiding = true;
        } else {
            guiding_field = new openpgl::cpp::Field(guiding_device, guiding_fieldConfig);
        }
    } else {
        guiding_field = new openpgl::cpp::Field(guiding_device, guiding_fieldConfig);
    }

    if (guideTraining)
        enableGuiding = true;

    guiding_sampleStorage = new openpgl::cpp::SampleStorage();

    guiding_threadPathSegmentStorage = new ThreadLocal<openpgl::cpp::PathSegmentStorage*>(
            [this]() { openpgl::cpp::PathSegmentStorage* pss = new openpgl::cpp::PathSegmentStorage();
                size_t maxPathSegments = this->maxDepth >= 1 ? this->maxDepth*2 : 30;
                pss->Reserve(maxPathSegments);
                pss->SetMaxDistance(guidingInfiniteLightDistance);
                return pss;});

    guiding_threadSurfaceSamplingDistribution = new ThreadLocal<openpgl::cpp::SurfaceSamplingDistribution*>(
            [this]() { return new openpgl::cpp::SurfaceSamplingDistribution(guiding_field); });

    guiding_threadVolumeSamplingDistribution = new ThreadLocal<openpgl::cpp::VolumeSamplingDistribution*>(
            [this]() { return new openpgl::cpp::VolumeSamplingDistribution(guiding_field); });

    Vector2i resolution = camera.GetFilm().PixelBounds().Diagonal();
    sensor = camera.GetFilm().GetPixelSensor();

    if (guideSettings.loadISGBuffer) {
        if (FileExists(guideSettings.isgBufferFileName)) {
            imageSpaceGuidingBuffer = new openpgl::cpp::util::ImageSpaceGuidingBuffer(guideSettings.isgBufferFileName);
            imageSpaceGuidingBufferReady = true;
            calculateImageSpaceGuidingBuffer = false;
        } else {
            std::cout << "Warning: ImageSpaceGuidingBuffer file does not exists: isgBufferFileName = " << guideSettings.isgBufferFileName << std::endl;
        }
    }

    if (!imageSpaceGuidingBufferReady && (guideSettings.storeISGBuffer || (guideSettings.guideVSP && guideSettings.guidePrimaryVSP) || guideSettings.guideRR)) {
        calculateImageSpaceGuidingBuffer = true;
        openpgl::cpp::util::ImageSpaceGuidingBuffer::Config cfg({resolution.x, resolution.y});
        if (guideSettings.guideRR) {
            cfg.EnableContributionEstimate(true);
        } else { 
            cfg.EnableContributionEstimate(false);
        }
        if (guideSettings.guideVSP && guideSettings.guidePrimaryVSP) {
            cfg.EnableVolumeScatterProbabilityEstimate(true);
            if (guideSettings.vspCriterion == EVariance) {
                cfg.SetVolumeScatterProbabilityType(PGLVSPTypes::EVariance);
            }
        } else {
            cfg.EnableVolumeScatterProbabilityEstimate(false);
        }
        imageSpaceGuidingBuffer = new openpgl::cpp::util::ImageSpaceGuidingBuffer(cfg);
    }

    if (guideSettings.loadTrBuffer) {
        if (FileExists(guideSettings.trBufferFileName)) {
            trBuffer = new TrBuffer(guideSettings.trBufferFileName);
            trBufferLoad = true;
            calculateTrBuffer = false;
        } else {
            std::cout << "Warning: Tr buffer file does not exists: trBufferFileName = " << guideSettings.trBufferFileName << std::endl;
        }
    }

    if (!trBufferLoad && (guideSettings.storeTrBuffer || (guideSettings.guideVSP && guideSettings.guidePrimaryVSP && guideSettings.guideVSPSamplingMethod ==  EVillemin && guideSettings.collisionProbabilityBias))) {
        calculateTrBuffer = true;
        trBuffer = new TrBuffer(resolution);
    }

    if (guideSettings.guideRR) {
        this->minRRDepth = 1;
    }
}

GuidedVolPathVSPGIntegrator::~GuidedVolPathVSPGIntegrator() {
    //~RayIntegrator();

    if (enableGuiding) {
        openpgl::cpp::FieldStatistics surfaceStats = guiding_field->GetSurfaceStatistics();
        std::cout << "Surface Guiding Field Statistics: "<< std::endl << surfaceStats.ToString() << std::endl;
        openpgl::cpp::FieldStatistics volumeStats = guiding_field->GetVolumeStatistics();
        std::cout << "Volume Guiding Field Statistics: "<< std::endl << volumeStats.ToString() << std::endl;
    }

    if (guideSettings.storeGuidingCache) {
        std::cout << "GuidedVolPathVSPGIntegrator storing guiding cache = " << guideSettings.guidingCacheFileName << std::endl;
        guiding_field->Store(guideSettings.guidingCacheFileName);
    }

    if (guideSettings.storeISGBuffer){
        imageSpaceGuidingBuffer->Store(guideSettings.isgBufferFileName);
    }

    if (guideSettings.storeTrBuffer){
        trBuffer->Store(guideSettings.trBufferFileName);
    }

    delete guiding_device;
    delete guiding_sampleStorage;
    delete guiding_field;
    delete imageSpaceGuidingBuffer;
    delete trBuffer;
}

void GuidedVolPathVSPGIntegrator::PostProcessWave() {

    waveCounter++;
    std::cout << "GuidedVolPathVSPGIntegrator::PostProcessWave()" << std::endl;
    if (guideTraining) {
        const size_t numValidSamples = guiding_sampleStorage->GetSizeSurface() + guiding_sampleStorage->GetSizeVolume();
        std::cout << "Guiding Iteration: "<< guiding_field->GetIteration() << "\t numValidSamples: " << numValidSamples << "\t surfaceSamples: " << guiding_sampleStorage->GetSizeSurface() << "\t surfaceInvalidSamples: " << guiding_sampleStorage->GetSizeZeroValueSurface() << "\t volumeSamples: " << guiding_sampleStorage->GetSizeVolume() << "\t volumeInvalidSamples: " << guiding_sampleStorage->GetSizeZeroValueVolume() << std::endl;
        if (numValidSamples > 128) {
            Timer guidingFiledUpdateTimer;
            guiding_field->Update(*guiding_sampleStorage);
            guidingCacheUpdateTime += guidingFiledUpdateTimer.ElapsedSeconds();
            if (guiding_field->GetIteration() >= guideSettings.guideNumTrainingWaves) {
                guideTraining = false;
            }
            guiding_sampleStorage->Clear();
        }
    }

    guiding_sampleStorage->Clear();

    if (waveCounter == std::pow(2.0f, bufferWave)) {
        if (calculateImageSpaceGuidingBuffer) {
            Timer isgBufferTimer;
            imageSpaceGuidingBuffer->Update();
            isgBufferUpdateTime += isgBufferTimer.ElapsedSeconds();
            imageSpaceGuidingBufferReady = true;
        }

        bufferWave++;
    }
}

SampledSpectrum GuidedVolPathVSPGIntegrator::Li(Point2i pPixel, RayDifferential ray, SampledWavelengths &lambda,
                                            Sampler sampler, ScratchBuffer &scratchBuffer,
                                            VisibleSurface *visibleSurf) const {

    openpgl::cpp::PathSegmentStorage* pathSegmentStorage = guiding_threadPathSegmentStorage->Get();
    openpgl::cpp::SurfaceSamplingDistribution* surfaceSamplingDistribution = guiding_threadSurfaceSamplingDistribution->Get();
    openpgl::cpp::VolumeSamplingDistribution* volumeSamplingDistribution = guiding_threadVolumeSamplingDistribution->Get();

    openpgl::cpp::PathSegment* pathSegmentData = nullptr;

    openpgl::cpp::util::ImageSpaceGuidingBuffer::Sample isgbSample;

    SampledSpectrum pixelContributionEstimate(1.f);
    SampledSpectrum adjointEstimate(1.f);
    bool guideRR = false;
    const bool guideSurfaceRR = guideSettings.guideSurfaceRR;
    const bool guideVolumeRR = guideSettings.guideVolumeRR;
    if (guideSettings.guideRR && imageSpaceGuidingBufferReady) {
        openpgl::cpp::Vector3f pgPixelContributionEstimate = imageSpaceGuidingBuffer->GetContributionEstimate(openpgl::cpp::Point2i(pPixel[0], pPixel[1]));
        pixelContributionEstimate[0] = pgPixelContributionEstimate.x;
        pixelContributionEstimate[1] = pgPixelContributionEstimate.y;
        pixelContributionEstimate[2] = pgPixelContributionEstimate.z;
        guideRR = true;
    }

    GuidedBSDF gbsdf(&sampler, guiding_field, surfaceSamplingDistribution, guideSettings.guideSurface, guideSettings.guideSecondaryVSP, guideSettings.surfaceGuidingType);
    GuidedPhaseFunction gphase(&sampler, guiding_field, volumeSamplingDistribution, guideSettings.guideVolume, guideSettings.guideSecondaryVSP, guideSettings.volumeGuidingType);
    GuidedInscatteredRadiance ginscatteredradiance(guiding_field, volumeSamplingDistribution, false);
    float rr_correction = 1.0f;
    float misPDF = 1.0f;

    // Declare state variables for volumetric path sampling
    SampledSpectrum L(0.f), beta(1.f), r_u(1.f), r_l(1.f);
    bool specularBounce = false, anyNonSpecularBounces = false;
    int depth = 0;
    Float etaScale = 1;

    bool lastVertexVolume = false;

    SampledSpectrum bsdfWeight(1.f);
    bool add_direct_contribution = false;
    Float w = 0.f;

    LightSampleContext prevIntrContext;
    
    int channelIdx = lambda.ChannelIdx();

    while (true) {
        // Sample segment of volumetric scattering path
        PBRT_DBG("%s\n", StringPrintf("Path tracer depth %d, current L = %s, beta = %s\n",
                                      depth, L, beta)
                .c_str())
        pstd::optional<ShapeIntersection> si = Intersect(ray);
        Float tMax = si ? si->tHit : Infinity;

        SampledSpectrum transmittanceWeight = SampledSpectrum(1.0f);
        if (ray.medium && !std::isinf(tMax)) {
            // Sample the participating medium
            bool scattered = false, terminated = false;

            // Initialize _RNG_ for sampling the majorant transmittance
            uint64_t hash0 = Hash(sampler.Get1D());
            uint64_t hash1 = Hash(sampler.Get1D());
            RNG rng(hash0, hash1);

            // The pointer pathSegmentData is updated in SampleDistance, and we want the updated pointer address
            // Therefore we need to pass a pointer to pathSegmentData (i.e., a double pointer)
            openpgl::cpp::PathSegment** pathSegmentDataPointer = &pathSegmentData;
            SampleDistance(pPixel, ray, tMax, lambda, sampler, rng,
                           scattered, terminated, depth,
                           L, beta, r_u, r_l,
                           specularBounce, anyNonSpecularBounces, prevIntrContext,
                           lastVertexVolume,
                           pathSegmentStorage, pathSegmentDataPointer,
                           gbsdf, gphase, ginscatteredradiance, rr_correction,
                           transmittanceWeight,
                           isgbSample, guideRR, guideVolumeRR,
                           adjointEstimate, pixelContributionEstimate);
            pathSegmentData = *pathSegmentDataPointer;

            // Handle terminated, scattered, and unscattered medium rays
            if (terminated || !beta || !r_u)
                break;
            if (scattered)
                continue;
        }

        // Handle surviving unscattered rays
        guiding_addTransmittanceWeight(pathSegmentData, transmittanceWeight, lambda, colorSpace);

        // Add emitted light at volume path vertex or from the environment
        if (!si) {
            // Accumulate contributions from infinite light sources
            for (const auto &light : infiniteLights) {
                SampledSpectrum Le = light.Le(ray, lambda);
                if (light.Type() == LightType::DeltaDirection && depth != 0)
                    Le = SampledSpectrum(0.f);
                if (depth == 0 || specularBounce) {
                    L += beta * Le / r_u.Average();
                    guiding_addInfiniteLightEmission(pathSegmentStorage, guidingInfiniteLightDistance, ray, Le, 1.0f, lambda, colorSpace);
                } else {
                    // Add infinite light contribution using both PDFs with MIS
                    Float lightPDF = lightSampler.PMF(prevIntrContext, light) *
                                     light.PDF_Li(prevIntrContext, ray.d, true);
                    r_l *= lightPDF;
                    Float w_b = useNEE ? 1.0f / (r_u + r_l).Average() : 1.f;
                    L += beta * w_b * Le;
                    guiding_addInfiniteLightEmission(pathSegmentStorage, guidingInfiniteLightDistance, ray, Le, w_b, lambda, colorSpace);
                }
            }

            break;
        }
        // Incorporate emission from surface hit by ray
        SurfaceInteraction &isect = si->intr;
        SampledSpectrum Le = isect.Le(-ray.d, lambda);
        if (Le) {
            // Add contribution of emission from intersected surface
            if (depth == 0 || specularBounce) {
                L += beta * Le / r_u.Average();

                w = 1.0f;
                add_direct_contribution = true;
            } else {
                // Add surface light contribution using both PDFs with MIS
                Light areaLight(isect.areaLight);
                Float lightPDF = lightSampler.PMF(prevIntrContext, areaLight) *
                                 areaLight.PDF_Li(prevIntrContext, ray.d, true);
                r_l *= lightPDF;
                // TODO add handling of survivial probability
                Float w_l = useNEE ? 1.0f / (r_u + r_l).Average() : 1.0f;
                L += beta * w_l * Le;
                w = w_l;
                add_direct_contribution = true;
            }
        }

        // Get BSDF and skip over medium boundaries
        BSDF bsdf = isect.GetBSDF(ray, lambda, camera, scratchBuffer, sampler);
        if (!bsdf) {
            isect.SkipIntersection(&ray, si->tHit);
            continue;
        }

        pathSegmentData = guiding_newSurfacePathSegment(pathSegmentStorage, ray, si);
        transmittanceWeight = SampledSpectrum(1.f);

        if (add_direct_contribution)
        {
            guiding_addSurfaceEmission(pathSegmentData, Le, w, lambda, colorSpace);
        }
        add_direct_contribution = false;

        // Initialize _visibleSurf_ at first intersection
        if (depth == 0 && (visibleSurf || calculateImageSpaceGuidingBuffer)) {
            // Estimate BSDF's albedo
            // Define sample arrays _ucRho_ and _uRho_ for reflectance estimate
            constexpr int nRhoSamples = 16;
            const Float ucRho[nRhoSamples] = {
                    0.75741637, 0.37870818, 0.7083487, 0.18935409, 0.9149363, 0.35417435,
                    0.5990858,  0.09467703, 0.8578725, 0.45746812, 0.686759,  0.17708716,
                    0.9674518,  0.2995429,  0.5083201, 0.047338516};
            const Point2f uRho[nRhoSamples] = {
                    Point2f(0.855985, 0.570367), Point2f(0.381823, 0.851844),
                    Point2f(0.285328, 0.764262), Point2f(0.733380, 0.114073),
                    Point2f(0.542663, 0.344465), Point2f(0.127274, 0.414848),
                    Point2f(0.964700, 0.947162), Point2f(0.594089, 0.643463),
                    Point2f(0.095109, 0.170369), Point2f(0.825444, 0.263359),
                    Point2f(0.429467, 0.454469), Point2f(0.244460, 0.816459),
                    Point2f(0.756135, 0.731258), Point2f(0.516165, 0.152852),
                    Point2f(0.180888, 0.214174), Point2f(0.898579, 0.503897)};

            SampledSpectrum albedo = bsdf.rho(isect.wo, ucRho, uRho);

            if (visibleSurf)
                *visibleSurf = VisibleSurface(isect, albedo, lambda);

            RGB rgbAlbedo = albedo.ToRGB(lambda, *colorSpace);
            isgbSample.albedo = {rgbAlbedo[0], rgbAlbedo[1], rgbAlbedo[2]};
            isgbSample.normal = {isect.n[0], isect.n[1], isect.n[2]};
            isgbSample.SetSurfaceEvent(true);
        }

        // Terminate path if maximum depth reached
        if (depth++ >= maxDepth)
            break;

        ++surfaceInteractions;
        // Possibly regularize the BSDF
        if (regularize && anyNonSpecularBounces) {
            ++regularizedBSDFs;
            bsdf.Regularize();
        }

        // Guiding - Check if we can use guiding. If so intialize the guiding distribution
        Float v = sampler.Get1D();
        gbsdf.init(&bsdf, ray, si, v);
        if (guideRR && guideSurfaceRR) {
#ifdef OPENPGL_RADIANCE_CACHES
            adjointEstimate = gbsdf.OutgoingRadiance(-ray.d);
#endif
        }

        Float survivalProb = 1.f;
        if (guideRR && depth > minRRDepth) {
            if(guideSurfaceRR) {
                survivalProb = specularBounce ? 0.95 : openpgl::cpp::util::GuidedRussianRoulette(OPGLVector3f(beta), OPGLVector3f(adjointEstimate), OPGLVector3f(pixelContributionEstimate), 0.1f);
            } else {
                survivalProb = 1.f;
            }
        }

        if (depth == 1 && visibleSurf && guiding_field->GetIteration() > 0) {
            visibleSurf->guidingData.id = gbsdf.getId();
        }

        // Sample illumination from lights to find attenuated path contribution
        if (useNEE && IsNonSpecular(bsdf.Flags())) {
            SampledSpectrum Ld = SampleLd(isect, &gbsdf, nullptr, 1.0f, lambda, sampler, r_u);
            L += beta * Ld;
            DCHECK(IsInf(L.y(lambda)) == false);

            // Guiding - add scattered contribution from NEE
            guiding_addScatteredDirectLight(pathSegmentData, Ld, lambda, colorSpace);
        }
        prevIntrContext = LightSampleContext(isect);

        // Sample BSDF to get new path direction
        //Vector3f wo = isect.wo;  // Note isect.wo does an explicit Normalize step.
        Vector3f wo = -ray.d; // Use -ray.d to be on par to GuidedPath
        Float u = sampler.Get1D();
        pstd::optional<BSDFSample> bs = gbsdf.Sample_f(wo, u, sampler.Get2D());
        if (!bs)
            break;

        lastVertexVolume = false;

        rr_correction *= bs->pdf / bs->bsdfPdf;
        misPDF = bs->misPdf;
        // Update _beta_ and rescaled path probabilities for BSDF scattering
        bsdfWeight = bs->f * AbsDot(bs->wi, isect.shading.n) / bs->pdf;
        beta *= bsdfWeight;
        //if (bs->pdfIsProportional)
        //    r_l = r_u / bsdf.PDF(wo, bs->wi);
        //else
        //    r_l = r_u / bs->pdf;
        r_l = r_u / bs->misPdf;

        PBRT_DBG("%s\n", StringPrintf("Sampled BSDF, f = %s, pdf = %f -> beta = %s",
                                      bs->f, bs->pdf, beta)
                .c_str());
        DCHECK(IsInf(beta.y(lambda)) == false);
        // Update volumetric integrator path state after surface scattering
        specularBounce = bs->IsSpecular();
        anyNonSpecularBounces |= !bs->IsSpecular();
        if (bs->IsTransmission())
            etaScale *= Sqr(bs->eta);
        ray = isect.SpawnRay(ray, bsdf, bs->wi, bs->flags, bs->eta);

        // Account for attenuated subsurface scattering, if applicable
/*
        BSSRDF bssrdf = isect.GetBSSRDF(ray, lambda, camera, scratchBuffer);
        if (bssrdf && bs->IsTransmission()) {
            // Sample BSSRDF probe segment to find exit point
            Float uc = sampler.Get1D();
            Point2f up = sampler.Get2D();
            pstd::optional<BSSRDFProbeSegment> probeSeg = bssrdf.SampleSp(uc, up);
            if (!probeSeg)
                break;

            // Sample random intersection along BSSRDF probe segment
            uint64_t seed = MixBits(FloatToBits(sampler.Get1D()));
            WeightedReservoirSampler<SubsurfaceInteraction> interactionSampler(seed);
            // Intersect BSSRDF sampling ray against the scene geometry
            Interaction base(probeSeg->p0, ray.time, Medium());
            while (true) {
                Ray r = base.SpawnRayTo(probeSeg->p1);
                if (r.d == Vector3f(0, 0, 0))
                    break;
                pstd::optional<ShapeIntersection> si = Intersect(r, 1);
                if (!si)
                    break;
                base = si->intr;
                if (si->intr.material == isect.material)
                    interactionSampler.Add(SubsurfaceInteraction(si->intr), 1.f);
            }

            if (!interactionSampler.HasSample())
                break;

            // Convert probe intersection to _BSSRDFSample_
            SubsurfaceInteraction ssi = interactionSampler.GetSample();
            BSSRDFSample bssrdfSample =
                bssrdf.ProbeIntersectionToSample(ssi, scratchBuffer);
            if (!bssrdfSample.Sp || !bssrdfSample.pdf)
                break;

            // Update path state for subsurface scattering
            Float pdf = interactionSampler.SampleProbability() * bssrdfSample.pdf[0];
            beta *= bssrdfSample.Sp / pdf;
            r_u *= bssrdfSample.pdf / bssrdfSample.pdf[0];
            SurfaceInteraction pi = ssi;
            pi.wo = bssrdfSample.wo;
            prevIntrContext = LightSampleContext(pi);
            // Possibly regularize subsurface BSDF
            BSDF &Sw = bssrdfSample.Sw;
            anyNonSpecularBounces = true;
            if (regularize) {
                ++regularizedBSDFs;
                Sw.Regularize();
            } else
                ++totalBSDFs;

            // Account for attenuated direct illumination subsurface scattering
            L += SampleLd(pi, &Sw, lambda, sampler, beta, r_u);

            // Sample ray for indirect subsurface scattering
            Float u = sampler.Get1D();
            pstd::optional<BSDFSample> bs = Sw.Sample_f(pi.wo, u, sampler.Get2D());
            if (!bs)
                break;
            beta *= bs->f * AbsDot(bs->wi, pi.shading.n) / bs->pdf;
            r_l = r_u / bs->pdf;
            // Don't increment depth this time...
            DCHECK(!IsInf(beta.y(lambda)));
            specularBounce = bs->IsSpecular();
            ray = RayDifferential(pi.SpawnRay(bs->wi));
        }
*/
        // Possibly terminate volumetric path with Russian roulette
        if (!beta)
            break;
        //SampledSpectrum rrBeta = beta * etaScale / r_u.Average();
        //        PBRT_DBG("%s\n",
        //         StringPrintf("etaScale %f -> rrBeta %s", etaScale, rrBeta).c_str());
        if (!guideRR && depth > minRRDepth) {
            const SampledSpectrum rrThroughputWeight = (beta / r_u.Average()) * rr_correction * etaScale;
            survivalProb = specularBounce ? 0.95 : openpgl::cpp::util::StandardThroughputBasedRussianRoulette(OPGLVector3f(rrThroughputWeight));
        }
        if (survivalProb < 1 && depth > minRRDepth) {
            Float q = std::max<Float>(0, 1 - survivalProb);
            if (sampler.Get1D() < q)
                break;
            beta /= 1 - q;
        }
        // Guiding - Add BSDF data to the current path segment
        guiding_addSurfaceData(pathSegmentData, bsdfWeight, bs->wi, bs->eta, bs->sampledRoughness, bs->pdf, survivalProb, lambda, colorSpace);
    }

    pathLength << depth;

    if(calculateImageSpaceGuidingBuffer)
    {
#if defined(PBRT_RGB_RENDERING)
        RGB colorRGB = L.ToRGB(lambda, *colorSpace);
#else
        RGB colorRGB = sensor->ToSensorRGB(L, lambda);
#endif
        isgbSample.contribution = {colorRGB[0], colorRGB[1], colorRGB[2]};
        imageSpaceGuidingBuffer->AddSample({pPixel[0], pPixel[1]}, isgbSample);
    }

    if (guideTraining)
    {
        //pathSegmentStorage->ValidateSegments();
        pathSegmentStorage->PropagateSamples(guiding_sampleStorage, true, true);
        pathSegmentStorage->Clear();
    }
    else
    {
        pathSegmentStorage->Clear();
    }
    return L;
}

void GuidedVolPathVSPGIntegrator::SampleDistance(Point2i pPixel, RayDifferential &ray, Float tMax,
                                                 SampledWavelengths &lambda, Sampler &sampler, RNG &rng,
                                                 bool &scattered, bool &terminated, int &depth,
                                                 SampledSpectrum &L, SampledSpectrum &beta, SampledSpectrum &r_u, SampledSpectrum &r_l,
                                                 bool &specularBounce, bool &anyNonSpecularBounces, LightSampleContext &prevIntrContext,
                                                 bool &lastVertexVolume,
                                                 openpgl::cpp::PathSegmentStorage* pathSegmentStorage,
                                                 openpgl::cpp::PathSegment** pathSegmentDataPointer,
                                                 const GuidedBSDF &gbsdf, GuidedPhaseFunction &gphase,
                                                 GuidedInscatteredRadiance ginscatteredradiance,
                                                 float rr_correction,
                                                 SampledSpectrum &transmittanceWeight,
                                                 openpgl::cpp::util::ImageSpaceGuidingBuffer::Sample &isgbSample,
                                                 bool guideRR, bool guideVolumeRR,
                                                 SampledSpectrum &adjointEstimate, SampledSpectrum &pixelContributionEstimate) const {
    int channelIdx = lambda.ChannelIdx();
    
    // Retrieve the target VSP value from the data structure for the primary or secondary ray
    bool guideScatterDecision = false;
    Float vsp = -1.f;
    if (depth == 0) {
        if (guideSettings.guideVSP && guideSettings.guidePrimaryVSP)
            vsp = GetPrimaryRayVolumeScatterProbability(pPixel, guideScatterDecision);
    }
    else {
        if (guideSettings.guideVSP && guideSettings.guideSecondaryVSP) {
            if (lastVertexVolume)
                vsp = GetSecondaryRayVolumeScatterProbability(gphase, ray.d, guideScatterDecision);
            else
                vsp = GetSecondaryRayVolumeScatterProbability(gbsdf, ray.d, guideScatterDecision);
        }
    }

    if (guideScatterDecision)
        vsp = std::max(std::min(vsp, 0.999f), 0.001f);

    int densityQueryCountPerSegment = 0;

    // Two routines: resampling & delta tracking
    // The resampling routine: the standard resampling weights achieve the same sample distribution as delta tracking; our method modifies them to achieve VSP guiding
    // The delta tracking routine: can modify the sample distribution through adjusting distance sampling PDF and the real/null-collision probabilities
    
    bool use_resampling = guideSettings.guideVSPSamplingMethod == EResampling && !ray.medium.IsHomogeneous();
    // Homogeneous volume: can do analytical VSPG, always enters the delta tracking routine
    // Heterogeneous volume: NDS & NDS+ -> delta tracking routine; resampling -> the resampling routine
    if (use_resampling) {
        // The resampling routine
        Float volumeRatioCompensated, majorantScale;
        Float weightSum = 0;
        SampledSpectrum trRatioEst(1.f), beta_resampling(1.f), r_u_resampling(1.f);
        CandidateData selectedCandidate;

        SampledSpectrum T_maj = SampleT_maj_Resampling(ray, tMax, sampler.Get1D(), rng, lambda,
                guideScatterDecision, vsp, volumeRatioCompensated, majorantScale,
                [&](Point3f p, MediumProperties mp, SampledSpectrum sigma_maj, SampledSpectrum T_maj) {
                    ++ densityQueryCountPerSegment;
                    // For each volume sample, compute the standard resampling weights (achieve the delta tracking distribution)
                    SampledSpectrum sigma_t = mp.sigma_s + mp.sigma_a;
                    SampledSpectrum sigma_n = ClampZero(sigma_maj - sigma_t);
                    Float wi = (sigma_t / sigma_maj * trRatioEst)[channelIdx];
                    Float sigmaTTrEstScalar = wi;

                    if (wi > 0) {
                        weightSum += wi;
                        // Reservoir sampling (he new volume sample VS the previous volume sample)
                        if (sampler.Get1D() < wi / weightSum) {
                            Float pdf = T_maj[channelIdx] * sigma_t[channelIdx];
                            SampledSpectrum throughputNumerator = beta_resampling *
                                                                  T_maj * mp.sigma_s / pdf;
                            SampledSpectrum throughputDenominator = r_u_resampling *
                                                                    T_maj * sigma_t / pdf;
                            selectedCandidate = CandidateData(p, mp, wi, sigmaTTrEstScalar,
                                                              throughputNumerator,
                                                              throughputDenominator);
                        }
                    }

                    Float pdf = T_maj[channelIdx] * sigma_n[channelIdx];
                    beta_resampling *= T_maj * sigma_n / pdf;
                    r_u_resampling *= T_maj * sigma_n / pdf;
                    trRatioEst *= sigma_n / sigma_maj;
                    return true;
                });

        beta_resampling *= T_maj / T_maj[channelIdx];
        r_u_resampling *= T_maj / T_maj[channelIdx];

        densityQueryCount << densityQueryCountPerSegment;

        // Record the ratio tracking estimate to store in the transmittance buffer
        if (depth == 0 && calculateTrBuffer)
            trBuffer->AddSample(pPixel, trRatioEst.ToRGB(lambda, *colorSpace));

        Float trRatioEstScalar = trRatioEst[channelIdx];
        CandidateData surfaceCandidate(ray(tMax), MediumProperties(),
                             trRatioEstScalar, trRatioEstScalar, beta_resampling, r_u_resampling);

        // VSP guiding through adjusting the resampling weights
        // Only adjust the surface resampling weights, an equivalent but simplified version of Eq. (24)
        if (guideScatterDecision && trRatioEstScalar < 1 && trRatioEstScalar > 0 && weightSum > 0) {
            Float trEstForScale = trRatioEstScalar;

            // Defensive resampling (i.e., MIS between VSP guiding and delta tracking)
            Float volRatio = volumeRatioCompensated * guideSettings.vspMISRatio
                             + (1 - trEstForScale) * (1 - guideSettings.vspMISRatio);
            Float surfRatio = 1 - volRatio;

            surfaceCandidate.wi = surfRatio / volRatio * weightSum;
        }

        weightSum += surfaceCandidate.wi;

        bool selectSurface = false;
        if (weightSum == 0) {
            return;
        }
        // Reservoir sampling (the surface sample VS the volume sample)
        else if (sampler.Get1D() < surfaceCandidate.wi / weightSum) {
            selectedCandidate = surfaceCandidate;
            selectSurface = true;
        }

        Float resamplingFactorScalar = weightSum * selectedCandidate.sigmaTTrEst / selectedCandidate.wi;
        
        if (selectSurface) {
            // Select the surface event = pass through the volume
            beta *= selectedCandidate.throughputNumerator * resamplingFactorScalar;
            r_u *= selectedCandidate.throughputDenominator;

            if (beta.HasNaNs() || r_u.HasNaNs() || IsInf(beta.y(lambda)) || IsInf(r_u.y(lambda))) {
                std::cout << "Surface candidate: " << depth << " " << beta << " " << r_u << " " << weightSum << " " << selectedCandidate.wi << " " << selectedCandidate.sigmaTTrEst << " " << std::endl;
                terminated = true;
                return;
            }
        }
        else {
            // Select the volume event = continue scattering inside the volume
            Point3f p = selectedCandidate.p;
            MediumProperties mp = selectedCandidate.mp;

            if (depth == 0) {
                SampledSpectrum albedo = mp.sigma_s / (mp.sigma_s + mp.sigma_a);
                RGB rgbAlbedo = albedo.ToRGB(lambda, *colorSpace);
                isgbSample.albedo = {rgbAlbedo[0], rgbAlbedo[1], rgbAlbedo[2]};
                isgbSample.normal = {-ray.d[0], -ray.d[1], -ray.d[1]};
                isgbSample.SetSurfaceEvent(false);
            }

            if (depth++ >= maxDepth) {
                terminated = true;
                return;
            }

            beta *= selectedCandidate.throughputNumerator * resamplingFactorScalar;
            r_u *= selectedCandidate.throughputDenominator;
            if (beta.HasNaNs() || r_u.HasNaNs() || IsInf(beta.y(lambda)) || IsInf(r_u.y(lambda))) {
                std::cout << "Volume candidate: " << depth << " " << beta << " " << r_u << " " << weightSum << " " << selectedCandidate.wi << " " << selectedCandidate.sigmaTTrEst << " " << std::endl;
                terminated = true;
                return;
            }

            transmittanceWeight *= selectedCandidate.throughputNumerator
                                   * resamplingFactorScalar / selectedCandidate.throughputDenominator;
            guiding_addTransmittanceWeight(*pathSegmentDataPointer, transmittanceWeight, lambda, colorSpace);
            *pathSegmentDataPointer = guiding_newVolumePathSegment(pathSegmentStorage, p, -ray.d);
            transmittanceWeight = SampledSpectrum(1.f);

            if (beta && r_u) {
                // Sample direct lighting at volume-scattering event
                MediumInteraction intr(p, -ray.d, ray.time, ray.medium,
                                       mp.phase);

                Float v = sampler.Get1D();
                gphase.init(&intr.phase, p, ray.d, v);
                if (guideRR && guideVolumeRR) {
#ifdef OPENPGL_RADIANCE_CACHES
                    adjointEstimate = gphase.InscatteredRadiance(-ray.d, true);
#endif
                }

                // calculate survival property
                Float survivalProb = 1.0f;
                if (depth > minRRDepth) {
                    if (guideRR) {
                        if(guideVolumeRR){
                            survivalProb = specularBounce ? 0.95 : openpgl::cpp::util::GuidedRussianRoulette(OPGLVector3f(beta), OPGLVector3f(adjointEstimate), OPGLVector3f(pixelContributionEstimate), 0.1f);
                        } else {
                            survivalProb = 1.f;
                        }
                    } else {
                        const SampledSpectrum rrThroughputWeight = (beta / r_u.Average()) * rr_correction;
                        survivalProb = specularBounce ? 0.95 : openpgl::cpp::util::StandardThroughputBasedRussianRoulette(OPGLVector3f(rrThroughputWeight));        
                    }
                }

                // Preform next-event estimation before RR
                if (useNEE) {
                    SampledSpectrum Ld = SampleLd(intr, nullptr, &gphase, 1.0f, lambda, sampler, r_u);
                    L += beta * Ld;

                    // Guiding - add scattered contribution from NEE
                    guiding_addScatteredDirectLight(*pathSegmentDataPointer, Ld, lambda, colorSpace);
                }

                // Perform stochastic path termination (Russian Roulette)
                if (survivalProb < 1 && depth > minRRDepth) {
                    Float q = std::max<Float>(0, 1 - survivalProb);
                    if (sampler.Get1D() < q){
                        terminated = true;
                        return;
                    }
                    beta /= 1 - q;
                }

                // Sample new direction at real-scattering event
                Point2f u = sampler.Get2D();
                pstd::optional<PhaseFunctionSample> ps =
                        gphase.Sample_p(-ray.d, u);
                if (!ps || ps->pdf == 0) {
                    terminated = true;
                    return;
                }
                else {
                    // Update ray path state for indirect volume scattering
                    Float phaseFunctionWeight = ps->p / ps->pdf;
                    beta *= phaseFunctionWeight;
                    r_l = r_u / ps->pdf;
                    prevIntrContext = LightSampleContext(intr);
                    scattered = true;
                    ray.o = p;
                    ray.d = ps->wi;
                    specularBounce = false;
                    anyNonSpecularBounces = true;

                    guiding_addVolumeData(*pathSegmentDataPointer, phaseFunctionWeight, ps->wi, ps->pdf, ps->meanCosine, survivalProb);

                    lastVertexVolume = true;
                }
            }
        }
    }
    else {
        // The delta tracking routine
        SampledSpectrum beta_factor(1.f), r_u_factor(1.f);
        SampledSpectrum T_maj = SampleT_maj_OpticalDepthSpace(ray, tMax, sampler.Get1D(), rng, lambda,
                                                              guideScatterDecision, vsp, guideSettings.vspMISRatio,
                                                              guideSettings.guideVSPSamplingMethod == EVillemin,
                                                              beta_factor, r_u_factor,
                                            [&](Point3f p, MediumProperties mp, SampledSpectrum sigma_maj, SampledSpectrum T_maj, bool activateNDS=false) {
                    ++ densityQueryCountPerSegment;

                    // Handle medium scattering event for ray
                    if (!beta) {
                        terminated = true;
                        return false;
                    }
                    ++volumeInteractions;
                    // Add emission from medium scattering event
                    if (depth < maxDepth && mp.Le) {
                        // Compute $\beta'$ at new path vertex
                        Float pdf = sigma_maj[channelIdx] * T_maj[channelIdx];
                        SampledSpectrum betap = beta * T_maj / pdf;

                        // Compute rescaled path probability for absorption at path vertex
                        SampledSpectrum r_e = r_u * sigma_maj * T_maj / pdf;

                        // Update _L_ for medium emission
                        if (r_e)
                            L += betap * mp.sigma_a * mp.Le / r_e.Average();
                    }

                    // Compute medium event probabilities for interaction
#if defined(VOLUME_ABSORB)
                    Float pAbsorb = mp.sigma_a[channelIdx] / sigma_maj[channelIdx];
                    Float pScatter = mp.sigma_s[channelIdx] / sigma_maj[channelIdx];
                    Float pNull = std::max<Float>(0, 1 - pAbsorb - pScatter);

                    CHECK_GE(1 - pAbsorb - pScatter, -1e-6);
                    // Sample medium scattering event type and update path
                    Float um = rng.Uniform<Float>();
                    int mode = SampleDiscrete({pAbsorb, pScatter, pNull}, um);
                    if (mode == 0) {
                        // Handle absorption along ray path>
                        terminated = true;
                        return false;

                    } else if (mode == 1) {
#else
                    SampledSpectrum sigma_t = mp.sigma_s + mp.sigma_a;
                    SampledSpectrum albedo = mp.sigma_s / sigma_t;
                    Float pScatter = sigma_t[channelIdx] / sigma_maj[channelIdx];

                    bool VilleminCollisionProbabilityBias = false;
                    if (depth == 0 && guideSettings.guideVSPSamplingMethod == EVillemin && guideSettings.collisionProbabilityBias && trBufferLoad && activateNDS) {
                        // NDS+: adjust the real/null-collision probability
                        VilleminCollisionProbabilityBias = true;
                        // Requires pre-computed transmittance estimate
                        // Therefore only enabled for the primary ray
                        Float trEstCache = trBuffer->GetTransmittance(pPixel)[channelIdx];
                        Float gamma = 1 + trEstCache;
                        pScatter = pow(pScatter, 1 / gamma);
                    }

                    CHECK_GE(1 - pScatter, -1e-6);

                    Float pNull = std::max<Float>(0, 1 - pScatter);

                    // Sample medium scattering event type and update path
                    Float um = rng.Uniform<Float>();
                    int mode = SampleDiscrete({pScatter, pNull}, um);
                    if (mode == 0) {
#endif

                        if (depth == 0) {
                            SampledSpectrum albedo = mp.sigma_s / (mp.sigma_s + mp.sigma_a);

                            RGB rgbAlbedo = albedo.ToRGB(lambda, *colorSpace);
                            isgbSample.albedo = {rgbAlbedo[0], rgbAlbedo[1], rgbAlbedo[2]};
                            isgbSample.normal = {-ray.d[0], -ray.d[1], -ray.d[1]};
                            isgbSample.SetSurfaceEvent(false);
                        }

                        // Handle scattering along ray path
                        // Stop path sampling if maximum depth has been reached
                        if (depth++ >= maxDepth) {
                            terminated = true;
                            return false;
                        }

                        // Update _beta_ and _r_u_ for real-scattering event
#if defined(VOLUME_ABSORB)
                        Float pdf = T_maj[channelIdx] * mp.sigma_s[channelIdx];
                        beta *= T_maj * mp.sigma_s / pdf;
                        r_u *= T_maj * mp.sigma_s / pdf;
#else
                        Float pdf = T_maj[channelIdx] * sigma_t[channelIdx];
                        beta *= T_maj * mp.sigma_s / pdf;
                        r_u *= T_maj * sigma_t / pdf;
                        if (VilleminCollisionProbabilityBias)
                            r_u *= sigma_maj * pScatter / sigma_t;
#endif
                        transmittanceWeight *= (T_maj * mp.sigma_s) / pdf;

                        beta *= beta_factor;
                        r_u *= r_u_factor;
                        transmittanceWeight *= beta_factor / r_u_factor[channelIdx];

                        guiding_addTransmittanceWeight(*pathSegmentDataPointer, transmittanceWeight, lambda, colorSpace);
                        *pathSegmentDataPointer = guiding_newVolumePathSegment(pathSegmentStorage, p, -ray.d);
                        transmittanceWeight = SampledSpectrum(1.f);

                        if (beta && r_u) {
                            // Sample direct lighting at volume-scattering event
                            MediumInteraction intr(p, -ray.d, ray.time, ray.medium,
                                                   mp.phase);

                            Float v = sampler.Get1D();
                            gphase.init(&intr.phase, p, ray.d, v);
                            if (guideRR && guideVolumeRR) {
#ifdef OPENPGL_RADIANCE_CACHES
                                adjointEstimate = gphase.InscatteredRadiance(-ray.d, true);
#endif
                            }

                            // calculate survival property
                            Float survivalProb = 1.0f;
                            if (depth > minRRDepth) {
                                if (guideRR) {
                                    if(guideVolumeRR){
                                        survivalProb = specularBounce ? 0.95 : openpgl::cpp::util::GuidedRussianRoulette(OPGLVector3f(beta), OPGLVector3f(adjointEstimate), OPGLVector3f(pixelContributionEstimate), 0.1f);
                                    } else {
                                        survivalProb = 1.f;
                                    }
                                } else {
                                    const SampledSpectrum rrThroughputWeight = (beta / r_u.Average()) * rr_correction;
                                    survivalProb = specularBounce ? 0.95 : openpgl::cpp::util::StandardThroughputBasedRussianRoulette(OPGLVector3f(rrThroughputWeight));        
                                }
                            }

                            // Preform next-event estimation before RR
                            if (useNEE){
                                SampledSpectrum Ld = SampleLd(intr, nullptr, &gphase, 1.0f, lambda, sampler, r_u);
                                L += beta * Ld;

                                // Guiding - add scattered contribution from NEE
                                guiding_addScatteredDirectLight(*pathSegmentDataPointer, Ld, lambda, colorSpace);
                            }

                            // Perform stochastic path termination (Russian Roulette)
                            if (survivalProb < 1 && depth > minRRDepth) {
                                Float q = std::max<Float>(0, 1 - survivalProb);
                                if (sampler.Get1D() < q){
                                    terminated = true;
                                    return false;
                                }
                                beta /= 1 - q;
                            }

                            // Continue path
                            // Sample new direction at real-scattering event
                            Point2f u = sampler.Get2D();
                            pstd::optional<PhaseFunctionSample> ps =
                                    gphase.Sample_p(-ray.d, u);
                            if (!ps || ps->pdf == 0)
                                terminated = true;
                            else {
                                // Update ray path state for indirect volume scattering
                                Float phaseFunctionWeight = ps->p / ps->pdf;
                                beta *= phaseFunctionWeight;
                                r_l = r_u / ps->pdf;
                                prevIntrContext = LightSampleContext(intr);
                                scattered = true;
                                ray.o = p;
                                ray.d = ps->wi;
                                specularBounce = false;
                                anyNonSpecularBounces = true;

                                guiding_addVolumeData(*pathSegmentDataPointer, phaseFunctionWeight, ps->wi, ps->pdf, ps->meanCosine, survivalProb);

                                lastVertexVolume = true;
                            }
                        }
                        return false;

                    } else {
                        // Handle null scattering along ray path
                        SampledSpectrum sigma_n =
                                ClampZero(sigma_maj - mp.sigma_a - mp.sigma_s);
                        Float pdf = T_maj[channelIdx] * sigma_n[channelIdx];
                        beta *= T_maj * sigma_n / pdf;
                        transmittanceWeight *= T_maj * sigma_n / pdf;
                        if (pdf == 0) {
                            beta = SampledSpectrum(0.f);
                            transmittanceWeight = SampledSpectrum(0.f);
                        }
                        r_u *= T_maj * sigma_n / pdf;
                        if (VilleminCollisionProbabilityBias)
                            r_u *= sigma_maj * (1 - pScatter) / sigma_n;
                        r_l *= T_maj * sigma_maj / pdf;
                        return beta && r_u;
                    }
                });

        bool multiply_T_maj = !(scattered || terminated || !beta || !r_u);
        if (multiply_T_maj) {
            beta *= T_maj / T_maj[channelIdx];
            r_u *= T_maj / T_maj[channelIdx];
            r_l *= T_maj / T_maj[channelIdx];
            transmittanceWeight *= T_maj / T_maj[channelIdx];

            beta *= beta_factor;
            r_u *= r_u_factor;
            r_l *= r_u_factor;
            transmittanceWeight *= beta_factor / r_u_factor[channelIdx];
        }

        densityQueryCount << densityQueryCountPerSegment;
    }
    return;
}

inline Float GuidedVolPathVSPGIntegrator::GetPrimaryRayVolumeScatterProbability(const Point2i &pPixel,
                                                                                bool &scatterPrimary) const {
    Float vsp = -1.f;
    if (imageSpaceGuidingBuffer->IsReady()) {
        vsp = imageSpaceGuidingBuffer->GetVolumeScatterProbabilityEstimate({pPixel.x, pPixel.y});
    } else {
        vsp = 0.5f;
    }

    if (std::isnan(vsp) || vsp < 0.f || vsp > 1.f)
        scatterPrimary = false;
    else
        scatterPrimary = true;
    return vsp;
}

inline Float GuidedVolPathVSPGIntegrator::GetSecondaryRayVolumeScatterProbability(
        const GuidedPhaseFunction &gphase, Vector3f wi, bool &scatterSecondary) const {
    Float vsp = gphase.VolumeScatterProbability(wi);

    if (std::isnan(vsp) || vsp < 0.f || vsp > 1.f)
        scatterSecondary = false;
    else
        scatterSecondary = true;
    return vsp;
}

inline Float GuidedVolPathVSPGIntegrator::GetSecondaryRayVolumeScatterProbability(
        const GuidedBSDF &gbsdf, Vector3f wi, bool &scatterSecondary) const {
    Float vsp = gbsdf.VolumeScatterProbability(wi);

    if (std::isnan(vsp) || vsp < 0.f || vsp > 1.f)
        scatterSecondary = false;
    else
        scatterSecondary = true;
    return vsp;
}

SampledSpectrum GuidedVolPathVSPGIntegrator::SampleLd(const Interaction &intr, const GuidedBSDF *bsdf, const GuidedPhaseFunction *phase,
                                                  const Float survivalProb, SampledWavelengths &lambda, Sampler sampler,
                                                  SampledSpectrum r_p) const {
    int channelIdx = lambda.ChannelIdx();

    // Estimate light-sampled direct illumination at _intr_
    // Initialize _LightSampleContext_ for volumetric light sampling
    LightSampleContext ctx;
    if (bsdf) {
        ctx = LightSampleContext(intr.AsSurface());
        // Try to nudge the light sampling position to correct side of the surface
        BxDFFlags flags = bsdf->Flags();
        if (IsReflective(flags) && !IsTransmissive(flags))
            ctx.pi = intr.OffsetRayOrigin(intr.wo);
        else if (IsTransmissive(flags) && !IsReflective(flags))
            ctx.pi = intr.OffsetRayOrigin(-intr.wo);

    } else
        ctx = LightSampleContext(intr);

    // Sample a light source using _lightSampler_
    Float u = sampler.Get1D();
    pstd::optional<SampledLight> sampledLight = lightSampler.Sample(ctx, u);
    Point2f uLight = sampler.Get2D();
    if (!sampledLight)
        return SampledSpectrum(0.f);
    Light light = sampledLight->light;
    DCHECK(light && sampledLight->p != 0);

    // Sample a point on the light source
    pstd::optional<LightLiSample> ls = light.SampleLi(ctx, uLight, lambda, true);
    if (!ls || !ls->L || ls->pdf == 0)
        return SampledSpectrum(0.f);
    Float p_l = sampledLight->p * ls->pdf;

    // Evaluate BSDF or phase function for light sample direction
    Float scatterPDF;
    SampledSpectrum f_hat;
    Vector3f wo = intr.wo, wi = ls->wi;
    if (bsdf) {
        // Update _f_hat_ and _scatterPDF_ accounting for the BSDF
        f_hat = bsdf->f(wo, wi) * AbsDot(wi, intr.AsSurface().shading.n);
        scatterPDF = survivalProb * bsdf->PDF(wo, wi);

    } else {
        // Update _f_hat_ and _scatterPDF_ accounting for the phase function
        CHECK(intr.IsMediumInteraction());
        //PhaseFunction phase = intr.AsMedium().phase;
        f_hat = SampledSpectrum(phase->p(wo, wi));
        scatterPDF = survivalProb * phase->PDF(wo, wi);
    }
    if (!f_hat)
        return SampledSpectrum(0.f);

    // Declare path state variables for ray to light source
    Ray lightRay = intr.SpawnRayTo(ls->pLight);
    SampledSpectrum T_ray(1.f), r_l(1.f), r_u(1.f);
    RNG rng(Hash(lightRay.o), Hash(lightRay.d));

    while (lightRay.d != Vector3f(0, 0, 0)) {
        // Trace ray through media to estimate transmittance
        pstd::optional<ShapeIntersection> si = Intersect(lightRay, 1 - ShadowEpsilon);
        // Handle opaque surface along ray's path
        if (si && si->intr.material)
            return SampledSpectrum(0.f);
        // Update transmittance for current ray segment
        if (lightRay.medium) {
            Float tMax = si ? si->tHit : (1 - ShadowEpsilon);
            Float u = rng.Uniform<Float>();
            SampledSpectrum T_maj =
                    SampleT_maj(lightRay, tMax, u, rng, lambda,
                                [&](Point3f p, MediumProperties mp, SampledSpectrum sigma_maj,
                                    SampledSpectrum T_maj) {
                                    // Update ray transmittance estimate at sampled point
                                    // Update _T_ray_ and PDFs using ratio-tracking estimator
                                    SampledSpectrum sigma_n =
                                            ClampZero(sigma_maj - mp.sigma_a - mp.sigma_s);
                                    Float pdf = T_maj[channelIdx] * sigma_maj[channelIdx];
                                    T_ray *= T_maj * sigma_n / pdf;
                                    r_l *= T_maj * sigma_maj / pdf;
                                    r_u *= T_maj * sigma_n / pdf;

                                    // Possibly terminate transmittance computation using
                                    // Russian roulette
                                    SampledSpectrum Tr = T_ray / (r_l + r_u).Average();
                                    if (Tr.MaxComponentValue() < 0.05f) {
                                        Float q = 0.75f;
                                        if (rng.Uniform<Float>() < q)
                                            T_ray = SampledSpectrum(0.);
                                        else
                                            T_ray /= 1 - q;
                                    }

                                    if (!T_ray)
                                        return false;
                                    return true;
                                });
            // Update transmittance estimate for final segment
            T_ray *= T_maj / T_maj[channelIdx];
            r_l *= T_maj / T_maj[channelIdx];
            r_u *= T_maj / T_maj[channelIdx];
        }
        // Generate next ray segment or return final transmittance
        if (!T_ray)
            return SampledSpectrum(0.f);
        if (!si)
            break;
        lightRay = si->intr.SpawnRayTo(ls->pLight);
    }
    // Return path contribution function estimate for direct lighting
    r_l *= r_p * p_l;
    r_u *= r_p * scatterPDF;
    if (IsDeltaLight(light.Type()))
        return f_hat * T_ray * ls->L / r_l.Average();
    else
        return f_hat * T_ray * ls->L / (r_l + r_u).Average();
}

std::string GuidedVolPathVSPGIntegrator::ToString() const {
    return StringPrintf(
            "[ GuidedVolPathVSPGIntegrator maxDepth: %d lightSampler: %s regularize: %s ]", maxDepth,
            lightSampler, regularize);
}

std::unique_ptr<GuidedVolPathVSPGIntegrator> GuidedVolPathVSPGIntegrator::Create(
        const ParameterDictionary &parameters, const RGBColorSpace *colorSpace, Camera camera, Sampler sampler,
        Primitive aggregate, std::vector<Light> lights, const FileLoc *loc) {
    int maxDepth = parameters.GetOneInt("maxdepth", 5);
    int minRRDepth = parameters.GetOneInt("minrrdepth", 1);
    bool useNEE = parameters.GetOneBool("usenee", true);

    // Directional guiding parameters
    GuidingSettings guideSettings;
    guideSettings.guideSurface = parameters.GetOneBool("surfaceguiding", false);
    guideSettings.guideVolume = parameters.GetOneBool("volumeguiding", false);
    std::string strSurfaceGuidingType = parameters.GetOneString("surfaceguidingtype", "ris");
    guideSettings.surfaceGuidingType = strSurfaceGuidingType == "mis" ? EGuideMIS : EGuideRIS;
    std::string strVolumeGuidingType = parameters.GetOneString("volumeguidingtype", "mis");
    guideSettings.volumeGuidingType = strVolumeGuidingType == "mis" ? EGuideMIS : EGuideRIS;
    guideSettings.storeGuidingCache = parameters.GetOneBool("storeGuidingCache", false);
    guideSettings.loadGuidingCache = parameters.GetOneBool("loadGuidingCache", false);
    guideSettings.guidingCacheFileName = parameters.GetOneString("guidingCacheFileName", "");


    // VSP guiding parameters
    guideSettings.guideVSP = parameters.GetOneBool("vspguiding", false);
    guideSettings.guidePrimaryVSP = parameters.GetOneBool("vspprimaryguiding", true);
    guideSettings.guideSecondaryVSP = parameters.GetOneBool("vspsecondaryguiding", true);
    guideSettings.vspMISRatio = parameters.GetOneFloat("vspmisratio", 0.5f);

    std::string strVSPCreterion = parameters.GetOneString("vspcriterion", "Contribution");
    if(strVSPCreterion == "Contribution" || strVSPCreterion == "contribution") {
        guideSettings.vspCriterion = VSPCriterion::EContribution;
    } else if (strVSPCreterion == "Variance" || strVSPCreterion == "variance") {
        guideSettings.vspCriterion = VSPCriterion::EVariance;
    } 

    std::string strVSPSamplingMethod = parameters.GetOneString("vspsamplingmethod", "resampling");
    if(strVSPSamplingMethod == "Resampling" || strVSPSamplingMethod == "resampling") {
        guideSettings.guideVSPSamplingMethod = VSPSamplingMethodType::EResampling;
    } else if (strVSPSamplingMethod == "Villemin" || strVSPSamplingMethod == "villemin") {
        guideSettings.guideVSPSamplingMethod = VSPSamplingMethodType::EVillemin;
    }

    guideSettings.collisionProbabilityBias = parameters.GetOneBool("collisionProbabilityBias", false);


    // Image space buffer parameters (for primary ray VSPG)
    guideSettings.storeISGBuffer = parameters.GetOneBool("storeISGBuffer", false);
    guideSettings.loadISGBuffer = parameters.GetOneBool("loadISGBuffer", false);
    guideSettings.isgBufferFileName = parameters.GetOneString("isgBufferFileName", "");

    // Transmittance buffer parameters (for primary ray VSPG, used only in NDS+)
    guideSettings.storeTrBuffer = parameters.GetOneBool("storeTrBuffer", false);
    guideSettings.loadTrBuffer = parameters.GetOneBool("loadTrBuffer", false);
    guideSettings.trBufferFileName = parameters.GetOneString("trBufferFileName", "");

    // Guided RR parameters
    guideSettings.guideRR = parameters.GetOneBool("rrguiding", false);
    guideSettings.guideSurfaceRR = parameters.GetOneBool("surfacerrguiding", true);
    guideSettings.guideVolumeRR = parameters.GetOneBool("volumerrguiding", true);

    std::string lightStrategy = parameters.GetOneString("lightsampler", "bvh");
    bool regularize = parameters.GetOneBool("regularize", false);

    return std::make_unique<GuidedVolPathVSPGIntegrator>(maxDepth, minRRDepth, useNEE, guideSettings, colorSpace, camera, sampler, aggregate, lights, lightStrategy, regularize);
}
#endif

}  // namespace pbrt