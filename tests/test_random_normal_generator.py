# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import pytest
import numpy as np
from tacto.random_normal_generator import RandomNormalGenerator


@pytest.fixture()
def image_shape():
    return (320, 240, 3)


def test_random_normal_generator(image_shape):
    t = time.time()
    for i in range(100):
        np.random.normal(0, 7, size=image_shape)
    print(f"Took {time.time() - t} sec.")


def test_random_normal_generator_multiprocess(image_shape):
    r = RandomNormalGenerator(mean=0, std=7, size=image_shape, prefetch=100)

    time.sleep(2)
    t = time.time()
    for i in range(100):
        r.sample()
    print(f"Took {time.time() - t} sec.")
