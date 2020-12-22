# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import logging

import hydra
import numpy as np
import pybullet as p

import pybulletX as px

import tacto

from sawyer_gripper import SawyerGripper

log = logging.getLogger(__name__)


# Load the config YAML file from examples/conf/grasp.yaml
@hydra.main(config_path="conf", config_name="grasp")
def main(cfg):
    # Initialize digits
    digits = tacto.Sensor(**cfg.tacto)

    # Initialize World
    log.info("Initializing world")
    px.init()

    p.resetDebugVisualizerCamera(**cfg.pybullet_camera)

    robot = SawyerGripper(**cfg.sawyer_gripper)

    # [21, 24]
    digits.add_camera(robot.id, robot.digit_links)

    # Add object to pybullet and digit simulator
    obj = px.Body(**cfg.object)
    digits.add_body(obj)

    np.set_printoptions(suppress=True)

    # run p.stepSimulation in another thread
    t = px.utils.SimulationThread(real_time_factor=1.0)
    t.start()

    robot.reset()

    panel = px.gui.RobotControlPanel(robot)
    panel.start()

    while True:
        color, depth = digits.render()
        digits.updateGUI(color, depth)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
