# # Copyright (c) Facebook, Inc. and its affiliates.

# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

import deepdish as dd
import matplotlib.pyplot as plt
import numpy as np
from scipyplot.plot import rplot_data

logDir = "logs/test1"

optNameList = ["BO", "Random"]
# optNameList = ["BO"]
# optNameList = ["Random"]

X = []
maxLen = 50
select_epoch = 5
for optName in optNameList:
    x = []

    scores = []
    params = []

    for epoch in range(30):
        fn = "{}/{}_{}.h5".format(logDir, optName, epoch)
        try:
            data = dd.io.load(fn)
        except Exception as e:
            print(e)
            continue
        obj, param = data["obj"], data["param"]
        obj = obj[0]

        bestScore = 1e6
        bestParam = None

        for i in range(0, len(obj)):
            if obj[i] < bestScore:
                bestScore = obj[i]
                bestParam = param[:, i]

            if i == select_epoch:
                scores.append(bestScore.copy() / 8)
                params.append(bestParam.copy())

            if i >= 1:
                obj[i] = min(obj[i - 1], obj[i])

        x.append(obj[:maxLen])
        # print(obj)
    x = np.vstack(x) / 8
    X.append(x)


# print(X[0].shape)
# # Raw data
# x = []
# x.append(np.random.rand(100, 30))
# x.append(np.random.rand(50, 20) + 2)
# ---
fig = rplot_data(
    ratio="4:1",
    data=X,
    color=["C1", "C0"],
    legend=optNameList,
    distribution="median+68",
)

# plt.yscale("log")
plt.xlim([0, 50])
plt.ylim([0.5, 3.5])
# plt.yticks([0.6, 1, 2, 3, 4], [0.6, 1, 2, 3, 4], fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

plt.ylabel("Cost", fontsize=20)
plt.xlabel("Iterations", fontsize=20)

fig = plt.gcf()
# fig.set_size_inches((5, 2))
plt.tight_layout()
plt.show()
