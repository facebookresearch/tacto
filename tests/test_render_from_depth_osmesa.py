# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np  # noqa: E402
import tacto  # noqa: E402

os.environ["PYOPENGL_PLATFORM"] = "osmesa"


# Render from OSMesa should be deterministic
def test_render_from_depth_osmesa():
    # tacto.Renderer use np.random.randn to generate noise.
    # Fix the random seed here to make test deterministic
    np.random.seed(2020)

    # Create renderer
    renderer = tacto.Renderer(
        width=60, height=80, config_path=tacto.get_digit_config_path(), background=None
    )

    # Get the path of current file
    cur_path = os.path.dirname(__file__)

    # Render
    depthmap = np.load(os.path.join(cur_path, "depthmap.npy"))
    color_gt = np.load(os.path.join(cur_path, "color-osmesa-ground-truth.npy"))

    color, depth = renderer.render_from_depth(depthmap, scale=0.01)
    # NOTE(poweic): This is how ground-truth is generated. Run this in docker
    # and make sure you eyeball the resulting image before committing the change
    # np.save(os.path.join(cur_path, "color-osmesa-ground-truth.npy"), color)

    diff = color - color_gt
    rms = (diff ** 2).mean() ** 0.5

    atol = 0.0
    assert rms <= atol, f"Expect RMS < {atol} but got {rms}"
