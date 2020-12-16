# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import numpy as np
import tacto  # Import TACTO
from scipy import signal


def generate_ball(xyz):
    N = 100
    M = 150

    x = np.linspace(-1, 1, N)
    y = np.linspace(-1 / N * M, 1 / N * M, M)
    xv, yv = np.meshgrid(x, y)

    # Generate ball contact
    R = 0.25
    r = np.minimum(((xv - xyz[0]) ** 2 + (yv - xyz[1]) ** 2) ** 0.5, R)
    depthmap = (R ** 2 - r ** 2) ** 0.5

    # Smooth the surface
    kern = np.ones([5, 5])
    kern /= kern.sum()
    depthmap = signal.convolve2d(depthmap, kern, boundary="symm", mode="same")

    return depthmap


def main():

    # Create renderer
    renderer = tacto.Renderer(
        width=240,
        height=320,
        background=None,
        config_path=tacto.get_digit_config_path(),
    )

    # Render
    depthmap = generate_ball([0, 0])
    color, depth = renderer.render_from_depth(depthmap, scale=0.01)

    # Plot the imprints
    plt.subplot(121)
    plt.imshow(color)
    plt.subplot(122)
    plt.imshow(0.022 - depth, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
