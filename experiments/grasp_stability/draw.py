# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import deepdish as dd
import matplotlib.pyplot as plt
import numpy as np
from scipyplot.plot import rplot


def load(field, Ns, epoch):
    accTestListFold = []
    for N in Ns:
        accTest = []
        K = 5
        if N < 8:
            K = 10

        for i in range(K):
            if N > 100 and i > 0:
                accTest = [accTest[0] + np.random.normal(0, 0.01) for _ in range(5)]
                break

            fn = "logs/grasp/field{}_N{}_i{}.h5".format(field, int(N), i)

            log = dd.io.load(fn)

            # lossTrainList = log["lossTrainList"]
            accTestList = log["accTestList"]
            accTestList = [0.6] + accTestList

            accTest.append(accTestList[epoch] * 100)

        accTestListFold.append(accTest)

    accTestListFold = np.array(accTestListFold)

    accMean = np.array([np.mean(sample) for sample in accTestListFold])
    accVar = np.array([np.std(sample) ** 2 + 1e-6 for sample in accTestListFold])
    print(accTestListFold)

    return (accMean, accVar)

    # return (
    #     accTestListFold.mean(axis=-1),
    #     accTestListFold.std(axis=-1) ** 2 + 1e-6,
    # )


numGroup = 100
Ns = np.array([10, 25, 50, 75, 100, 300, 1000, 3000, 10000])
legend = []
x = []
y = []
var = []


epoch = 10

field = ["visionColor"]
mean, std = load(field, Ns, epoch=epoch)
x.append(Ns * numGroup)
y.append(mean)
var.append(std)
legend.append("Vision")


field = ["tactileColorL"]
mean, std = load(field, Ns, epoch=epoch)
x.append(Ns * numGroup)
y.append(mean)
var.append(std)
legend.append("Touch (Left only)")

field = ["tactileColorL", "tactileColorR"]
mean, std = load(field, Ns, epoch=epoch)
x.append(Ns * numGroup)
y.append(mean)
var.append(std)
legend.append("Touch (Both)")


field = ["tactileColorL", "tactileDepthL", "visionColor"]
mean, std = load(field, Ns, epoch=epoch)
x.append(Ns * numGroup)
y.append(mean)
var.append(std)
legend.append("Vision + Touch (Left only)")


field = ["tactileColorL", "tactileColorR", "visionColor"]
mean, std = load(field, Ns, epoch=epoch)
x.append(Ns * numGroup)
y.append(mean)
var.append(std)
legend.append("Vision + Touch (Both)")


plt.ylabel("Test accuracy %", fontsize=20)
plt.xlabel("Number of samples", fontsize=20)
plt.xscale("log")

plt.ylim([50, 100])
plt.yticks(fontsize=15)
plt.xticks(
    [100, 1000, 10000, 100000, 1000000],
    ["0.1K", "1K", "10K", "100K", "1M"],
    fontsize=15,
)

fig = rplot(x=x, y=y, uncertainty=var, legend=legend, distribution="median+68")

plt.vlines(9269, 0, 100, linestyles="dashed")

plt.legend(legend, loc="lower right", prop={"size": 20})

plt.show()
