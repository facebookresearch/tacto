# # Copyright (c) Facebook, Inc. and its affiliates.

# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

import numpy as np
from opto.opto.classes.OptTask import OptTask
import opto.utils as rutils
import RollingEnv


class Rolling(OptTask):
    def __init__(self, n_parameters=4, visualize=True):
        """
        Quadratic function
        """
        super(Rolling, self).__init__(
            f=self._f,
            fprime=self._g,
            name="Rolling",
            n_parameters=n_parameters,
            n_objectives=1,
            order=1,
            bounds=rutils.bounds(max=[2] * n_parameters, min=[-2] * n_parameters),
            task={"minimize"},
            labels_param=None,
            labels_obj=None,
            vectorized=False,
            info=None,
            opt_obj=0,
            opt_parameters=np.matrix([[0] * n_parameters]),
        )

        self.env = RollingEnv.RollingEnv(visTacto=visualize, visPyBullet=visualize)

    def _f(self, xs):
        costs = []
        goals = [
            [0.3, 0.3],
            [0.3, 0.5],
            [0.3, 0.7],
            [0.5, 0.3],
            [0.5, 0.7],
            [0.7, 0.3],
            [0.7, 0.5],
            [0.7, 0.7],
        ]
        print("xs", xs)
        for i in range(len(xs)):
            K = xs[i].reshape([2, 2]) / 1000

            c = 0
            for goal in goals:
                c += self.env.simulate(goal, K)

            costs.append([c / len(goals)])

        costs = np.matrix(costs)
        print("costs", costs)
        return costs

    def _g(self, x):
        return np.matrix(2 * x)


if __name__ == "__main__":
    env = Rolling()
    costs = env._f(np.array([[-0.1, 0, 0, -0.1], [-1, 0, 0, -1]]))
    print(costs)
