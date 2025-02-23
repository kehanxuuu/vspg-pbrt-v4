# Renderer code for the paper "Volume Scattering Probability Guiding"
### [Project Page](https://kehanxuuu.github.io/vspg-website/) | [Paper](https://kehanxuuu.github.io/vspg-website/static/pdfs/volume_scattering_probability_guiding_sa24.pdf) | [Scripts](https://github.com/kehanxuuu/vspg-rendering-scripts/) | [Scenes](https://drive.google.com/file/d/11mECG390H3CFszWaNu2i9QC87CDdTJAh/view?usp=sharing)

<img src="teaser.jpg" alt="teaser" width="1024"/>

This repository holds the renderer code for the SIGGRAPH ASIA 2024 paper "Volume Scattering Probability Guiding". For instructions on reproducing the experiments, please refer to the [scripts](https://github.com/kehanxuuu/vspg-rendering-scripts/) repository.

The volume scattering probability (VSP) guiding algorithm is implemented in [PBRT V4](https://github.com/mmp/pbrt-v4). The implementation is primarily contained in two files: `src/pbrt/cpu/guidedvolpathintegrator.cpp` and `src/pbrt/media_sampleTMaj.h`. The data structures for both primary and secondary ray VSP guiding are integrated into Intel's Open Path Guiding Library ([OpenPGL](https://github.com/RenderKit/openpgl)), which is included as a Git submodule. The logic for training and querying the spatial-directional radiance caching data structure with OpenPGL is handled in `guiding.h`. This data structure is used for both directional guiding and secondary ray VSP guiding.

# Build Instructions

```
git clone --recursive https://github.com/kehanxuuu/vspg-pbrt-v4.git
cd vspg-pbrt-v4
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../install -DGLFW_BUILD_WAYLAND=OFF ..
```
Tested on: Linux, Macbook M1.

For more detailed instructions and usage, refer to the original [PBRT README](https://github.com/mmp/pbrt-v4).

# Guided Volume Path Integrator
The *guidedvolpathintegrator* extends the built-in *volpathintegrator* with directional guiding and VSP guiding. The modified distance sampling logic is implemented in `GuidedVolPathVSPGIntegrator::SampleDistance`.

Below is a list of guiding-related integrator parameters along with their default values and descriptions.
### Directional guiding parameters
* **surfaceguiding, volumeguiding**: bool (default = true); activate surface/volume directional guiding.
* **surfaceguidingtype**: string, "mis" or "ris" (default = "ris"); the strategy for defensive sampling with BSDF sampling.
* **volumeguidingtype**: string, "mis" or "ris" (default = "mis"); the strategy for defensive sampling with Phase function sampling.
* **storeGuidingCache, loadGuidingCache**: bool (default = false); store/load the data structure for directional guiding.
* **guidingCacheFileName**: string (default =  ""); name of the guiding cache to store/load.

### VSP guiding parameters
* **vspguiding**: bool (default = true); activate VSP guiding.
* **vspprimaryguiding, vspsecondaryguiding**: bool (default = true); separately activate VSP guiding for primary/secondary rays.
* **vspmisratio**: float (default = 0.5); the ratio for MIS defensive sampling with delta tracking.
* **vspcriterion**: string, "contribution" or "variance" (default = "variance"); the criterion to determine the optimal VSP.
* **vspsamplingmethod**: string, "resampling" or "nds" (default = "resampling"); "resampling" is our proposed distance sampling algorithm, whereas "nds" (normalized distance sampling) is the baseline for comparison.
* **collisionProbabilityBias**: bool (default = false); activate NDS+ when vspsamplingmethod is set to "nds".

### Image space buffer parameters (for primary ray VSPG)
* **storeISGBuffer, loadISGBuffer**: bool (default = false); store/load the image space buffer.
* **isgBufferFileName**: string (default =  ""); name of the image space buffer to store/load.

### Transmittance buffer parameters (for primary ray VSPG, used only in NDS+)
* **storeTrBuffer, loadTrBuffer**: bool(default = false); store/load the transmittance for primary rays.
* **trBufferFileName**: string (default =  ""); name of the transmittance buffer to store/load.

The default parameters runs our VSP guiding algorithm with variance-based criterion, while enabling directional guiding. You can simply turn **vspguiding** on/off to compare between the effect of enabling/disabling VSP guiding. For more configurations (e.g., compare with the baseline), refer to the [testcase file](https://github.com/kehanxuuu/vspg-rendering-scripts/blob/vspg/pbrt-testcases/vspg.py) in the rendering scripts.

Note that both directional guiding and secondary ray VSP guiding rely on the same spatial-directional data structure implemented in OpenPGL. As a result, the training procedure is activated if either of these two features is enabled.