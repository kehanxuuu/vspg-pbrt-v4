/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2013 Intel Corporation. */

#ifndef PBRT_VSP_BUFFER_H
#define PBRT_VSP_BUFFER_H

#include <pbrt/denoiser.h>
#include <pbrt/util/image.h>
#include <pbrt/util/parallel.h>

#define USE_PVOL_EST
//#define USE_PVOL_CORRECTION
//#define DENOISE_AFTER_PRODUCT

namespace pbrt {

struct TrBuffer {
    TrBuffer(const Vector2i& resolution): resolution(resolution){
        int numPixels = resolution[0]*resolution[1];
        spp = new int[numPixels];
        transmittanceBuffer = new RGB[numPixels];

        pbrt::ParallelFor(
            0, numPixels,
            PBRT_CPU_GPU_LAMBDA(int i) {
                spp[i] = 0.f;
                transmittanceBuffer[i] = {0.f,0.f,0.f};
            });
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
