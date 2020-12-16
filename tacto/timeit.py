# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from functools import wraps
from collections import deque, defaultdict

import numpy as np


def timeit(f):

    WINDOW_SIZE = 128
    timeit._elapsed = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))

    def summarize():
        print("\33[33m----- Summarize -----\33[0m")
        for k, q in timeit._elapsed.items():
            print(f"{k:55s} took: {np.mean(q):.5f} sec [{len(q)} samples]")

    timeit.summarize = summarize

    @wraps(f)
    def wrap(*args, **kwargs):
        t = time.time()
        result = f(*args, **kwargs)
        elapsed = time.time() - t
        timeit._elapsed[repr(f)].append(elapsed)
        return result

    return wrap
