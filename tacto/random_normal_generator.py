# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from multiprocessing import Process, Queue


class RandomNormalGenerator(Process):
    def __init__(self, mean, std, size, prefetch=16):
        super().__init__(daemon=True)
        self.mean = mean
        self.std = std
        self.size = size

        self._q = Queue(maxsize=prefetch)
        self.start()

    def run(self):
        while True:
            noise = np.random.normal(self.mean, self.std, self.size)
            self._q.put(noise)

    def sample(self):
        return self._q.get()
