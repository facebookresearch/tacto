# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from tacto.timeit import timeit


@timeit
def foo(x):
    y = np.random.normal(0, 7, size=x.shape)
    return y


def test_timeit():
    s = 0
    x = np.zeros((320, 240, 3), dtype=np.uint8)
    for i in range(100):
        y = foo(x)
        s += y.sum()
    print(s)
