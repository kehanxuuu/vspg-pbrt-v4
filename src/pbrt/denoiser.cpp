// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/denoiser.h>

namespace pbrt {

Denoiser::Denoiser(Vector2i resolution, bool haveAlbedoAndNormal): resolution(resolution), haveAlbedoAndNormal(haveAlbedoAndNormal){}

#if defined(PBRT_WITH_OIDN)
OIDNDenoiser::OIDNDenoiser(Vector2i resolution, bool haveAlbedoAndNormal): Denoiser(resolution, haveAlbedoAndNormal){
    oidnDevice = oidn::newDevice();
    oidnDevice.commit();

    // Check for errors
	const char* errorMessage;
	//if (oidnDevice.getError(errorMessage) != oidn::Error::None)
	//	std::cout << "Error: " << errorMessage << std::endl;

    size_t numPixels = resolution[0] * resolution[1];

    bufferColor = oidnDevice.newBuffer(numPixels * 3 * sizeof(float));
	bufferAlbedo = oidnDevice.newBuffer(numPixels * 3 * sizeof(float));		
	bufferNormal = oidnDevice.newBuffer(numPixels * 3 * sizeof(float));

    bufferColorOutput = oidnDevice.newBuffer(numPixels * 3 * sizeof(float));
	
    if (filterFeatures) {
        bufferNormalOutput = oidnDevice.newBuffer(numPixels * 3 * sizeof(float));
	    bufferAlbedoOutput = oidnDevice.newBuffer(numPixels * 3 * sizeof(float));

        oidnAlbedoFilter = oidnDevice.newFilter("RT");
		oidnAlbedoFilter.setImage("albedo", bufferAlbedo, oidn::Format::Float3, resolution[0], resolution[1]);
		oidnAlbedoFilter.setImage("output", bufferAlbedoOutput, oidn::Format::Float3, resolution[0], resolution[1]);
		oidnAlbedoFilter.commit();
			
		oidnNormalFilter = oidnDevice.newFilter("RT");
		oidnNormalFilter.setImage("normal", bufferNormal, oidn::Format::Float3, resolution[0], resolution[1]);
		oidnNormalFilter.setImage("output", bufferNormalOutput, oidn::Format::Float3, resolution[0], resolution[1]);
		oidnNormalFilter.commit();

		oidnColorFilter = oidnDevice.newFilter("RT");
		oidnColorFilter.setImage("color", bufferColor, oidn::Format::Float3, resolution[0], resolution[1]);
		oidnColorFilter.setImage("albedo", bufferAlbedoOutput, oidn::Format::Float3, resolution[0], resolution[1]);
		oidnColorFilter.setImage("normal", bufferNormalOutput, oidn::Format::Float3, resolution[0], resolution[1]);
		oidnColorFilter.setImage("output", bufferColorOutput, oidn::Format::Float3, resolution[0], resolution[1]);
		oidnColorFilter.set("cleanAux", true); // auxiliary images will be prefiltered
		oidnColorFilter.set("hdr", true);
		oidnColorFilter.commit();
    } else {
        oidnColorFilter = oidnDevice.newFilter("RT");
		oidnColorFilter.setImage("color", bufferColor, oidn::Format::Float3, resolution[0], resolution[1]);
		oidnColorFilter.setImage("albedo", bufferAlbedo, oidn::Format::Float3, resolution[0], resolution[1]);
		oidnColorFilter.setImage("normal", bufferNormal, oidn::Format::Float3, resolution[0], resolution[1]);
		oidnColorFilter.setImage("output", bufferColorOutput, oidn::Format::Float3, resolution[0], resolution[1]);

		oidnColorFilter.set("hdr", true);
		oidnColorFilter.commit();
    }
    		// Check for errors
	//if (oidnDevice.getError(errorMessage) != oidn::Error::None)
	//	std::cout << "Error: " << errorMessage << std::endl;


}

void OIDNDenoiser::Denoise(RGB *rgb, Normal3f *n, RGB *albedo, RGB *result) {
    // Copy data to OIDN buffers
	bufferColor.write(0, resolution[0]*resolution[1]*3*sizeof(float), rgb);
	bufferNormal.write(0, resolution[0]*resolution[1]*3*sizeof(float), n);
	bufferAlbedo.write(0, resolution[0]*resolution[1]*3*sizeof(float), albedo);
    if (filterFeatures){
		oidnAlbedoFilter.execute();
		oidnNormalFilter.execute();
    }
    oidnColorFilter.execute();
    bufferColorOutput.read(0, resolution[0]*resolution[1]*3*sizeof(float), result);

    const char* errorMessage;
	//if (oidnDevice.getError(errorMessage) != oidn::Error::None)
	//	std::cout << "Error: " << errorMessage << std::endl;
}

#endif

}