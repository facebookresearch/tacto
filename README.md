# TACTO: A Fast, Flexible and Open-source Simulator for High-Resolution Vision-based Tactile Sensors

[![License: MIT](https://img.shields.io/github/license/facebookresearch/tacto)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/tacto)](https://pypi.org/project/tacto/)
[![CircleCI](https://circleci.com/gh/facebookresearch/tacto.svg?style=shield)](https://circleci.com/gh/facebookresearch/tacto)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<a href="https://digit.ml/">
<img height="20" src="/website/static/img/digit-logo.svg" alt="DIGIT-logo" />
</a>

<img src="/website/static/img/teaser.jpg?raw=true" alt="TACTO Simulator" />


This package provides a simulator for vision-based tactile sensors, such as [DIGIT](https://digit.ml).
It provides models for the integration with PyBullet, as well as a renderer of touch readings.
For more information refer to the corresponding paper [TACTO: A Fast, Flexible, and Open-source Simulator for High-resolution Vision-based Tactile Sensors](https://arxiv.org/abs/2012.08456).

NOTE: the simulator is not meant to provide a physically accurate dynamics of the contacts (e.g., deformation, friction), but rather relies on existing physics engines.

**For updates and discussions please join the #TACTO channel at the [www.touch-sensing.org](https://www.touch-sensing.org/) community.**


## Installation

The preferred way of installation is through PyPi:

```bash
pip install tacto
```

Alternatively, you can manually clone the repository and install the package using:

```bash
git clone https://github.com/facebookresearch/tacto.git
cd tacto
pip install -e .
```

## Content
This package contain several components:
1) A renderer to simulate readings from vision-based tactile sensors.
2) An API to simulate vision-based tactile sensors in PyBullet.
3) Mesh models and configuration files for the [DIGIT](https://digit.ml) and Omnitact sensors.

## Usage

Additional packages ([torch](https://github.com/pytorch/pytorch), [gym](https://github.com/openai/gym), [pybulletX](https://github.com/facebookresearch/pybulletX)) are required to run the following examples.
You can install them by `pip install -r requirements/examples.txt`.

For a basic example on how to use TACTO in conjunction with PyBullet look at [TBD],

For an example of how to use just the renderer engine look at [examples/demo_render.py](examples/demo_render.py).

For advanced examples of how to use the simulator with PyBullet look at the [examples folder](examples).

* [examples/demo_pybullet_digit.py](examples/demo_pybullet_digit.py): rendering RGB and Depth readings with a [DIGIT](https://digit.ml) sensor.
<img src="/website/static/img/demo_digit.gif?raw=true" alt="Demo DIGIT" />

* [examples/demo_pybullet_allegro_hand.py](examples/demo_pybullet_omnitact.py): rendering 4 DIGIT sensors on an Allegro Hand.
<img src="/website/static/img/demo_allegro.gif?raw=true" alt="Demo Allegro" />

* [examples/demo_pybullet_omnitact.py](examples/demo_pybullet_omnitact.py): rendering RGB and Depth readings with a [OmniTact](https://arxiv.org/pdf/2003.06965.pdf) sensor.
<img src="/website/static/img/demo_omnitact.gif?raw=true" alt="Demo OmniTact" />

* [examples/demo_pybullet_grasp.py](examples/demo_grasp.py): mounted on parallel-jaw grippers and grasping objects with different configurations.
<img src="/website/static/img/demo_grasp.gif?raw=true" alt="Demo Grasp" />

* [examples/demo_pybullet_rolling.py](examples/demo_rolling.py): rolling a marble with two DIGIT sensors.
<img src="/website/static/img/demo_rolling.gif?raw=true" alt="Demo Rolling" />

* [examples/demo_pybullet_digit_shadow.py](examples/demo_pybullet_digit_shadow.py): enable shadow rendering.
<img src="/website/static/img/demo_shadow.gif?raw=true" alt="Demo Shadow" />

### Headless Rendering

NOTE: the renderer requires a screen. For rendering headless, use the "EGL" mode with GPU and CUDA driver or "OSMESA" with CPU. 
See [PyRender](https://pyrender.readthedocs.io/en/latest/install/index.html) for more details.

Additionally, install the patched version of PyOpenGL via,

```
pip install git+https://github.com/mmatl/pyopengl.git@76d1261adee2d3fd99b418e75b0416bb7d2865e6
```

You may then specify which engine to use for headless rendering, for example,

```
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa" # osmesa cpu rendering
```

## Operating System
We recommend to conduct experiments on **Ubuntu**.

For **macOS**, there exists some visualization problem between pybullet.GUI and pyrender as we know of. Please let us know if it can be resolved, and we will share the information at the repo!

## License
This project is licensed under MIT license, as found in the [LICENSE](LICENSE) file.


## Citing
If you use this project in your research, please cite:

```BibTeX
@Article{Wang2022TACTO,
  author   = {Wang, Shaoxiong and Lambeta, Mike and Chou, Po-Wei and Calandra, Roberto},
  title    = {{TACTO}: A Fast, Flexible, and Open-source Simulator for High-resolution Vision-based Tactile Sensors},
  journal  = {IEEE Robotics and Automation Letters (RA-L)},
  year     = {2022},
  volume   = {7},
  number   = {2},
  pages    = {3930--3937},
  issn     = {2377-3766},
  doi      = {10.1109/LRA.2022.3146945},
  url      = {https://arxiv.org/abs/2012.08456},
}
```

