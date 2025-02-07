# Renderer code for the paper "Volume Scattering Probability Guiding"
### [Project Page](https://kehanxuuu.github.io/vspg-website/) | [Paper](https://kehanxuuu.github.io/vspg-website/static/pdfs/volume_scattering_probability_guiding_sa24.pdf) | [Scripts](https://github.com/kehanxuuu/vspg-rendering-scripts/) | [Scenes](https://github.com/kehanxuuu/vspg-scenes/)

<img src="teaser.jpg" alt="teaser" width="1024"/>

This repository holds the renderer code for the SIGGRAPH ASIA 2024 paper "Volume Scattering Probability Guiding". The algorithm is implemented in [PBRT V4](https://github.com/mmp/pbrt-v4); please refer to its documentation for build instructions.

The implementation of volume scattering probability (VSP) guiding is primarily contained in two files: `src/pbrt/cpu/guidedvolpathintegrator.cpp` and `src/pbrt/media_sampleTMaj.h`. The data structures for both primary and secondary ray VSP guiding are integrated into Intel's Open Path Guiding Library ([OpenPGL](https://github.com/RenderKit/openpgl)), which is included as a Git submodule. The logic for training and querying the spatial-directional radiance caching data structure with OpenPGL is handled in `guiding.h`. This data structure is used for both directional guiding and secondary ray VSP guiding.

For instructions on reproducing the experiments, please refer to the [scripts](https://github.com/kehanxuuu/vspg-rendering-scripts/) repository.