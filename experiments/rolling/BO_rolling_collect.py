# # Copyright (c) Facebook, Inc. and its affiliates.

# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

import logging
import os
import time

import deepdish as dd
import opto
import opto.regression as rregression
import Rolling

# from opto.functions import *
from dotmap import DotMap
from opto.opto.acq_func import UCB

start = time.time()


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# To log file
fh = logging.FileHandler("example.log")
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


# task = Rolling.Rolling(visualize=True)
task = Rolling.Rolling(visualize=False)
stopCriteria = opto.opto.classes.StopCriteria(maxEvals=1)

objList = []
paramList = []
optNameList = ["BO", "Random"]

for epoch in range(0, 30):
    for optName in optNameList:
        print(optName, epoch)
        p = DotMap()
        p.verbosity = 1
        p.acq_func = UCB(model=[], logs=[], parameters={"alpha": 0.1})
        # p.acq_func = EI(model=None, logs=None)
        # p.optimizer = opto.CMAES
        p.visualize = True
        p.model = rregression.GP

        if optName == "BO":
            opt = opto.BO(parameters=p, task=task, stopCriteria=stopCriteria)
        elif optName == "Random":
            opt = opto.RandomSearch(parameters=p, task=task, stopCriteria=stopCriteria)
        try:
            opt.optimize()
        except:
            continue
        logs = opt.get_logs()

        obj = logs.get_objectives()
        param = logs.get_parameters()

        logDir = "logs/test1"
        os.makedirs(logDir, exist_ok=True)

        fn = "{}/{}_{}.h5".format(logDir, optName, epoch)
        dd.io.save(fn, {"obj": obj, "param": param})
