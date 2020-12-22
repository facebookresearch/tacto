# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import cv2
import hydra
import pybullet as p
import tacto  # Import TACTO

import pybulletX as px

log = logging.getLogger(__name__)


# Load the config YAML file from examples/conf/rolling.yaml
@hydra.main(config_path="conf", config_name="rolling")
def main(cfg):
    # Initialize digits
    bg = cv2.imread("conf/bg_digit_240_320.jpg")
    digits = tacto.Sensor(**cfg.tacto, background=bg)

    # Initialize World
    log.info("Initializing world")
    px.init()

    p.resetDebugVisualizerCamera(**cfg.pybullet_camera)

    # Create and initialize DIGIT URDF (top & bottom)
    digit_top = px.Body(**cfg.digits.top)
    digit_bottom = px.Body(**cfg.digits.bottom)

    digits.add_camera(digit_top.id, [-1])
    digits.add_camera(digit_bottom.id, [-1])

    # Add object to pybullet and tacto simulator
    obj = px.Body(**cfg.object)
    digits.add_body(obj)

    # Create control panel to control the 6DoF pose of the object
    panel = px.gui.PoseControlPanel(digit_top, **cfg.object_control_panel)
    panel.start()
    log.info("Use the slides to move the object until in contact with the DIGIT")

    # run p.stepSimulation in another thread
    t = px.utils.SimulationThread(real_time_factor=1.0)
    t.start()

    while True:
        (_, _, z), orientation = obj.get_base_pose()
        if z <= 0.01:
            obj.reset()
            digit_top.reset()

        color, depth = digits.render()
        digits.updateGUI(color, depth)

    t.stop()


if __name__ == "__main__":
    main()
