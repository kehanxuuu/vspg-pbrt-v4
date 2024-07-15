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
    std::cout<< "GuidedVolPathIntegrator:" <<std::endl;
    std::cout<< "\t maxDepth = " << maxDepth << std::endl;
    std::cout<< "\t minRRDepth = " << minRRDepth << std::endl;
    std::cout<< "\t useNEE = " << useNEE << std::endl;
    std::cout<< "\t surfaceGuiding = " << guideSettings.guideSurface << std::endl;
    std::cout<< "\t volumeGuiding = " << guideSettings.guideVolume << std::endl;
    std::cout<< "\t surfaceGuidingType = " << guideSettings.surfaceGuidingType << std::endl;
    std::cout<< "\t volumeGuidingType = " << guideSettings.volumeGuidingType << std::endl;
    std::cout<< "\t loadGuidingCache = " << guideSettings.loadGuidingCache << std::endl;
    std::cout<< "\t guidingCacheFileName = " << guideSettings.guidingCacheFileName << std::endl;
    std::cout<< "\t lightSampleStrategy = " << lightSampleStrategy << std::endl;
    std::cout<< "\t regularize = " << regularize << std::endl;

    guideTraining = guideSettings.guideSurface || guideSettings.guideVolume || guideSettings.guideSurfaceRR || guideSettings.guideVolumeRR;

    guiding_device = new openpgl::cpp::Device(PGL_DEVICE_TYPE_CPU_4);
    guiding_fieldConfig.Init(PGL_SPATIAL_STRUCTURE_KDTREE, PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM);

    if (guideSettings.loadGuidingCache) {
        if (FileExists(guideSettings.guidingCacheFileName)) {
            std::cout<< "GuidedVolPathIntegrator: loading guiding cache = "<< guideSettings.guidingCacheFileName <<std::endl;
            guiding_field = new openpgl::cpp::Field(guiding_device, guideSettings.guidingCacheFileName);
            guideTraining = false;
        } else {
            guiding_field = new openpgl::cpp::Field(guiding_device, guiding_fieldConfig);
        }
    } else {
        guiding_field = new openpgl::cpp::Field(guiding_device, guiding_fieldConfig);
    }
    guiding_sampleStorage = new openpgl::cpp::SampleStorage();

    guiding_threadPathSegmentStorage = new ThreadLocal<openpgl::cpp::PathSegmentStorage*>(
            [this]() { openpgl::cpp::PathSegmentStorage* pss = new openpgl::cpp::PathSegmentStorage(true);
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

    if (guideSettings.loadVSPBuffer) {
        if (FileExists(guideSettings.vspBufferFileName)) {
            vspBuffer = new VSPBuffer(guideSettings.vspBufferFileName);
            vspBufferReady = true;
            calulateVSPBuffer = false;
        } else {
            std::cout << "Warning: VSP buffer file does not exists: vspBufferFileName = " << guideSettings.vspBufferFileName << std::endl;
        }
    }

    if (!vspBufferReady && guideSettings.useVSPBuffer) {
        calulateVSPBuffer = true;
        vspBuffer = new VSPBuffer(resolution);
    }

    if (guideSettings.loadContributionEstimate) {
        if (FileExists(guideSettings.contributionEstimateFileName)) {
            contributionEstimate = new ContributionEstimate(guideSettings.contributionEstimateFileName);
            contributionEstimateReady = true;
            calulateContributionEstimate = false;
        } else {
            std::cout << "Warning: Contribution estimate file does not exists: contributionEstimateFileName = " << guideSettings.contributionEstimateFileName << std::endl;
        }
    }

    if (!contributionEstimateReady && (guideSettings.storeContributionEstimate || guideSettings.guideRR)){
        calulateContributionEstimate = true;
        contributionEstimate = new ContributionEstimate(resolution);
    }

    if (guideSettings.guideRR) {
        this->minRRDepth = 1;
    }
}


GuidedVolPathVSPGIntegrator::~GuidedVolPathVSPGIntegrator() {
    //~RayIntegrator();
    if (guideSettings.storeGuidingCache) {
        std::cout << "GuidedVolPathIntegrator storing guiding cache = " << guideSettings.guidingCacheFileName << std::endl;
        guiding_field->Store(guideSettings.guidingCacheFileName);
    }

    if (guideSettings.useVSPBuffer && guideSettings.storeVSPBuffer){
        vspBuffer->Store(guideSettings.vspBufferFileName);
    }

    if (guideSettings.storeContributionEstimate){
        contributionEstimate->Store(guideSettings.contributionEstimateFileName);
    }

    delete guiding_device;
    delete guiding_sampleStorage;
    delete guiding_field;
    delete contributionEstimate;
    delete vspBuffer;
}

void GuidedVolPathVSPGIntegrator::PostProcessWave() {

    waveCounter++;
    std::cout << "GuidedVolPathIntegrator::PostProcessWave()" << std::endl;
    if (guideTraining) {
        const size_t numValidSamples = guiding_sampleStorage->GetSizeSurface() + guiding_sampleStorage->GetSizeVolume();
        std::cout << "Guiding Iteration: "<< guiding_field->GetIteration() << "\t numValidSamples: " << numValidSamples << "\t surfaceSamples: " << guiding_sampleStorage->GetSizeSurface() << "\t volumeSamples: " << guiding_sampleStorage->GetSizeVolume() << std::endl;
        if (numValidSamples > 128) {
            guiding_field->Update(*guiding_sampleStorage);
            if (guiding_field->GetIteration() >= guideSettings.guideNumTrainingWaves) {
                guideTraining = false;
            }
            guiding_sampleStorage->Clear();
        }
    }

    guiding_sampleStorage->Clear();

    if (calulateVSPBuffer && waveCounter == std::pow(2.0f, vspBufferWave)) {
        vspBuffer->Update();
        vspBufferReady = true;
        vspBufferWave++;
    }

    if (calulateContributionEstimate && waveCounter == std::pow(2.0f, contributionEstimateWave)) {
        contributionEstimate->Update();
        contributionEstimateReady = true;
        contributionEstimateWave++;
    }
}

SampledSpectrum GuidedVolPathVSPGIntegrator::Li(Point2i pPixel, RayDifferential ray, SampledWavelengths &lambda,
                                            Sampler sampler, ScratchBuffer &scratchBuffer,
                                            VisibleSurface *visibleSurf) const {

    openpgl::cpp::PathSegmentStorage* pathSegmentStorage = guiding_threadPathSegmentStorage->Get();
    openpgl::cpp::SurfaceSamplingDistribution* surfaceSamplingDistribution = guiding_threadSurfaceSamplingDistribution->Get();
    openpgl::cpp::VolumeSamplingDistribution* volumeSamplingDistribution = guiding_threadVolumeSamplingDistribution->Get();

    openpgl::cpp::PathSegment* pathSegmentData = nullptr;

    ContributionEstimate::ContributionEstimateData ced;

    VSPBuffer::Sample vspSample;

    SampledSpectrum pixelContributionEstimate(1.f);
    SampledSpectrum adjointEstimate(1.f);
    bool guideRR = false;
    const bool guideSurfaceRR = guideSettings.guideSurfaceRR;
    const bool guideVolumeRR = guideSettings.guideVolumeRR;
    if (guideSettings.guideRR && contributionEstimateReady) {
        pixelContributionEstimate = contributionEstimate->GetContributionEstimate(pPixel);
        guideRR = true;
    }

    // Declare state variables for volumetric path sampling
    SampledSpectrum L(0.f), beta(1.f), r_u(1.f), r_l(1.f);
    bool specularBounce = false, anyNonSpecularBounces = false;
    int depth = 0;
    Float etaScale = 1;

    GuidedBSDF gbsdf(&sampler, guiding_field, surfaceSamplingDistribution, guideSettings.guideSurface, guideSettings.surfaceGuidingType);
    GuidedPhaseFunction gphase(&sampler, guiding_field, volumeSamplingDistribution, guideSettings.guideVolume, guideSettings.volumeGuidingType);
    float rr_correction = 1.0f;
    float misPDF = 1.0f;

    SampledSpectrum bsdfWeight(1.f);
    bool add_direct_contribution = false;
    Float w = 0.f;

    LightSampleContext prevIntrContext;

    while (true) {
        Float survivalProb = 1.f;
        // Sample segment of volumetric scattering path
        PBRT_DBG("%s\n", StringPrintf("Path tracer depth %d, current L = %s, beta = %s\n",
                                      depth, L, beta)
                .c_str());
        pstd::optional<ShapeIntersection> si = Intersect(ray);

        SampledSpectrum transmittanceWeight = SampledSpectrum(1.0f);
        if (ray.medium) {
            // Sample the participating medium
            bool scattered = false, terminated = false;
            Float tMax = si ? si->tHit : Infinity;
            // Initialize _RNG_ for sampling the majorant transmittance
            uint64_t hash0 = Hash(sampler.Get1D());
            uint64_t hash1 = Hash(sampler.Get1D());
            RNG rng(hash0, hash1);

            SampledSpectrum T_maj = SampleT_maj(
                    ray, tMax, sampler.Get1D(), rng, lambda,
                    [&](Point3f p, MediumProperties mp, SampledSpectrum sigma_maj,
                        SampledSpectrum T_maj) {
                        // Handle medium scattering event for ray
                        if (!beta) {
                            terminated = true;
                            return false;
                        }
                        ++volumeInteractions;
                        // Add emission from medium scattering event
                        if (depth < maxDepth && mp.Le) {
                            // Compute $\beta'$ at new path vertex
                            Float pdf = sigma_maj[lambda.ChannelIdx()] * T_maj[lambda.ChannelIdx()];
                            SampledSpectrum betap = beta * T_maj / pdf;

                            // Compute rescaled path probability for absorption at path vertex
                            SampledSpectrum r_e = r_u * sigma_maj * T_maj / pdf;

                            // Update _L_ for medium emission
                            if (r_e)
                                L += betap * mp.sigma_a * mp.Le / r_e.Average();
                        }

                        // Compute medium event probabilities for interaction
#if defined(VOLUME_ABSORB)
                        Float pAbsorb = mp.sigma_a[lambda.ChannelIdx()] / sigma_maj[lambda.ChannelIdx()];
                    Float pScatter = mp.sigma_s[lambda.ChannelIdx()] / sigma_maj[lambda.ChannelIdx()];
                    Float pNull = std::max<Float>(0, 1 - pAbsorb - pScatter);

                    CHECK_GE(1 - pAbsorb - pScatter, -1e-6);
                    // Sample medium scattering event type and update path
                    Float um = rng.Uniform<Float>();
                    int mode = SampleDiscrete({pAbsorb, pScatter, pNull}, um);
                    if (mode == 0) {
                        // Handle absorption along ray path
                        terminated = true;
                        return false;

                    } else if (mode == 1) {
#else
                        SampledSpectrum sigma_t = mp.sigma_s + mp.sigma_a;
                        SampledSpectrum albedo = mp.sigma_s / sigma_t;
                        Float pScatter = sigma_t[lambda.ChannelIdx()] / sigma_maj[lambda.ChannelIdx()];
                        Float pNull = std::max<Float>(0, 1 - pScatter);

                        CHECK_GE(1 - pScatter, -1e-6);
                        // Sample medium scattering event type and update path
                        Float um = rng.Uniform<Float>();
                        int mode = SampleDiscrete({pScatter, pNull}, um);
                        if (mode == 0) {
#endif

                            if (depth == 0) {
                                SampledSpectrum albedo = mp.sigma_s / (mp.sigma_s + mp.sigma_a);

                                vspSample.albedo = albedo.ToRGB(lambda, *colorSpace);
                                vspSample.normal = Normal3f(-ray.d);
                                vspSample.isVolume = true;

                                ced.albedo = albedo.ToRGB(lambda, *colorSpace);
                                ced.normal = Normal3f(-ray.d);
                            }

                            // Handle scattering along ray path
                            // Stop path sampling if maximum depth has been reached
                            if (depth++ >= maxDepth) {
                                terminated = true;
                                return false;
                            }

                            // Update _beta_ and _r_u_ for real-scattering event
#if defined(VOLUME_ABSORB)
                            Float pdf = T_maj[lambda.ChannelIdx()] * mp.sigma_s[lambda.ChannelIdx()];
                        beta *= T_maj * mp.sigma_s / pdf;
                        r_u *= T_maj * mp.sigma_s / pdf;
#else
                            Float pdf = T_maj[lambda.ChannelIdx()] * sigma_t[lambda.ChannelIdx()];
                            beta *= T_maj * mp.sigma_s / pdf;
                            r_u *= T_maj * sigma_t / pdf;
#endif
                            transmittanceWeight *= (T_maj * mp.sigma_s) / pdf;
                            guiding_addTransmittanceWeight(pathSegmentData, transmittanceWeight, lambda, colorSpace);
                            pathSegmentData = guiding_newVolumePathSegment(pathSegmentStorage, p, -ray.d);

                            if (beta && r_u) {
                                // Sample direct lighting at volume-scattering event
                                MediumInteraction intr(p, -ray.d, ray.time, ray.medium,
                                                       mp.phase);

                                Float v = sampler.Get1D();
                                gphase.init(&intr.phase, p, ray.d, v);
                                if (guideRR && guideVolumeRR) {
                                    adjointEstimate = gphase.InscatteredRadiance(-ray.d);
                                }

                                // calculate survival property
                                survivalProb = 1.0f;
                                if (depth > minRRDepth) {
                                    if (guideRR) {
                                        if (guideVolumeRR)
                                            survivalProb = specularBounce ? 0.95 : GuidedRussianRouletteProbability(beta, adjointEstimate, pixelContributionEstimate);
                                        else
                                            survivalProb = 1.f;
                                    } else {
                                        survivalProb = specularBounce ? 0.95 : StandardRussianRouletteSurvivalProbability((beta / r_u.Average()) * rr_correction, 1.0f);
                                    }
                                }

                                // Preform next-event estimation before RR
                                if (useNEE){
                                    SampledSpectrum Ld = SampleLd(intr, nullptr, &gphase, 1.0f, lambda, sampler, r_u);
                                    L += beta * Ld;

                                    // Guiding - add scattered contribution from NEE
                                    guiding_addScatteredDirectLight(pathSegmentData, Ld, lambda, colorSpace);
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

                                    guiding_addVolumeData(pathSegmentData, phaseFunctionWeight, ps->wi, ps->pdf, ps->meanCosine, survivalProb);
                                }
                            }
                            return false;

                        } else {
                            // Handle null scattering along ray path
                            SampledSpectrum sigma_n =
                                    ClampZero(sigma_maj - mp.sigma_a - mp.sigma_s);
                            Float pdf = T_maj[lambda.ChannelIdx()] * sigma_n[lambda.ChannelIdx()];
                            beta *= T_maj * sigma_n / pdf;
                            transmittanceWeight *= T_maj * sigma_n / pdf;
                            if (pdf == 0) {
                                beta = SampledSpectrum(0.f);
                                transmittanceWeight = SampledSpectrum(0.f);
                            }
                            r_u *= T_maj * sigma_n / pdf;
                            r_l *= T_maj * sigma_maj / pdf;
                            return beta && r_u;
                        }
                    });
            // Handle terminated, scattered, and unscattered medium rays
            if (terminated || !beta || !r_u)
                break;
            if (scattered)
                continue;

            transmittanceWeight *= T_maj / T_maj[lambda.ChannelIdx()];
            beta *= T_maj / T_maj[lambda.ChannelIdx()];
            r_u *= T_maj / T_maj[lambda.ChannelIdx()];
            r_l *= T_maj / T_maj[lambda.ChannelIdx()];
        }

        // Handle surviving unscattered rays
        guiding_addTransmittanceWeight(pathSegmentData, transmittanceWeight, lambda, colorSpace);

        // Add emitted light at volume path vertex or from the environment
        if (!si) {
            // Accumulate contributions from infinite light sources
            for (const auto &light : infiniteLights) {
                SampledSpectrum Le = light.Le(ray, lambda);
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

        if (add_direct_contribution)
        {
            guiding_addSurfaceEmission(pathSegmentData, Le, w, lambda, colorSpace);
        }
        add_direct_contribution = false;

        // Initialize _visibleSurf_ at first intersection
        if (depth == 0 && (visibleSurf || calulateVSPBuffer || calulateContributionEstimate)) {
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

            vspSample.albedo = albedo.ToRGB(lambda, *colorSpace);
            vspSample.normal = isect.n;
            vspSample.isVolume = false;

            ced.albedo = albedo.ToRGB(lambda, *colorSpace);
            ced.normal = isect.n;
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
            adjointEstimate = gbsdf.OutgoingRadiance(-ray.d);
        }

        if (guideRR && depth > minRRDepth) {
            if (guideSurfaceRR)
                survivalProb = specularBounce ? 0.95 : GuidedRussianRouletteProbability(beta, adjointEstimate, pixelContributionEstimate);
            else
                survivalProb = 1.f;
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
            survivalProb = specularBounce ? 0.95 : StandardRussianRouletteSurvivalProbability((beta / r_u.Average()) * rr_correction, etaScale);
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

    if(calulateVSPBuffer)
    {
#if defined(PBRT_RGB_RENDERING)
        vspSample.color = L.ToRGB(lambda, *colorSpace);
#else
        vspSample.color = sensor->ToSensorRGB(L, lambda);
#endif
        vspBuffer->AddSample(pPixel, vspSample);
    }

    if (calulateContributionEstimate)
    {
#if defined(PBRT_RGB_RENDERING)
        ced.color = L.ToRGB(lambda, *colorSpace);
#else
        ced.color = sensor->ToSensorRGB(L, lambda);
#endif
        contributionEstimate->Add(pPixel, ced);
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

SampledSpectrum GuidedVolPathVSPGIntegrator::SampleLd(const Interaction &intr, const GuidedBSDF *bsdf, const GuidedPhaseFunction *phase,
                                                  const Float survivalProb, SampledWavelengths &lambda, Sampler sampler,
                                                  SampledSpectrum r_p) const {
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
                                    Float pdf = T_maj[lambda.ChannelIdx()] * sigma_maj[lambda.ChannelIdx()];
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
            T_ray *= T_maj / T_maj[lambda.ChannelIdx()];
            r_l *= T_maj / T_maj[lambda.ChannelIdx()];
            r_u *= T_maj / T_maj[lambda.ChannelIdx()];
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
            "[ GuidedVolPathIntegrator maxDepth: %d lightSampler: %s regularize: %s ]", maxDepth,
            lightSampler, regularize);
}

std::unique_ptr<GuidedVolPathVSPGIntegrator> GuidedVolPathVSPGIntegrator::Create(
        const ParameterDictionary &parameters, const RGBColorSpace *colorSpace, Camera camera, Sampler sampler,
        Primitive aggregate, std::vector<Light> lights, const FileLoc *loc) {
    int maxDepth = parameters.GetOneInt("maxdepth", 5);
    int minRRDepth = parameters.GetOneInt("minrrdepth", 1);
    bool useNEE = parameters.GetOneBool("usenee", true);
    GuidingSettings guidingSettings;
    guidingSettings.knnLookup = parameters.GetOneBool("knnlookup", true);
    guidingSettings.guideSurface = parameters.GetOneBool("surfaceguiding", true);
    guidingSettings.guideVolume = parameters.GetOneBool("volumeguiding", true);
    guidingSettings.guideRR = parameters.GetOneBool("rrguiding", false);
    guidingSettings.guideSurfaceRR = parameters.GetOneBool("surfacerrguiding", true);
    guidingSettings.guideVolumeRR = parameters.GetOneBool("volumerrguiding", true);

    std::string strSurfaceGuidingType = parameters.GetOneString("surfaceguidingtype", "ris");
    guidingSettings.surfaceGuidingType = strSurfaceGuidingType == "mis" ? EGuideMIS : EGuideRIS;
    std::string strVolumeGuidingType = parameters.GetOneString("volumeguidingtype", "mis");
    guidingSettings.volumeGuidingType = strVolumeGuidingType == "mis" ? EGuideMIS : EGuideRIS;

    guidingSettings.storeGuidingCache = parameters.GetOneBool("storeGuidingCache", false);
    guidingSettings.loadGuidingCache = parameters.GetOneBool("loadGuidingCache", false);
    guidingSettings.guidingCacheFileName = parameters.GetOneString("guidingCacheFileName", "");

    // VSP buffer (screen space, for primary ray VSPG)
    guidingSettings.useVSPBuffer = parameters.GetOneBool("useVSPBuffer", false);
    guidingSettings.storeVSPBuffer = parameters.GetOneBool("storeVSPBuffer", false);
    guidingSettings.loadVSPBuffer = parameters.GetOneBool("loadVSPBuffer", false);
    guidingSettings.vspBufferFileName = parameters.GetOneString("vspBufferFileName", "");

    // Guided RR
    guidingSettings.storeContributionEstimate = parameters.GetOneBool("storeContributionEstimate", false);
    guidingSettings.loadContributionEstimate = parameters.GetOneBool("loadContributionEstimate", false);
    guidingSettings.contributionEstimateFileName = parameters.GetOneString("contributionEstimateFileName", "");

    std::string lightStrategy = parameters.GetOneString("lightsampler", "bvh");
    bool regularize = parameters.GetOneBool("regularize", false);

    return std::make_unique<GuidedVolPathVSPGIntegrator>(maxDepth, minRRDepth, useNEE, guidingSettings, colorSpace, camera, sampler, aggregate, lights, lightStrategy, regularize);
}
#endif

}  // namespace pbrt