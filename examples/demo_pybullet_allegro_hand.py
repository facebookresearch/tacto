# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import hydra
import pybullet as p
import tacto  # Import TACTO

import pybulletX as px

log = logging.getLogger(__name__)


# Load the config YAML file from examples/conf/allegro_hand.yaml
@hydra.main(config_path="conf", config_name="allegro_hand")
def main(cfg):
    # Initialize digits
    # FIXME(poweic): this has to be done before p.connect if PYOPENGL_PLATFORM set to "egl"
    digits = tacto.Sensor(**cfg.tacto)

    # Initialize World
    log.info("Initializing world")
    px.init()

    p.resetDebugVisualizerCamera(**cfg.pybullet_camera)

    # Add allegro hand
    allegro = px.Body(**cfg.allegro)

    # Add cameras to tacto simulator
    digits.add_camera(allegro.id, cfg.digit_link_id_allegro)

    # Add object to pybullet and tacto simulator
    obj = px.Body(**cfg.object)
    digits.add_body(obj)

    # Create control panel to control the 6DoF pose of the object
    panel = px.gui.PoseControlPanel(obj, **cfg.object_control_panel)
    panel.start()

    log.info("Use the slides to move the object until in contact with the DIGIT")

    # run p.stepSimulation in another thread
    t = px.utils.SimulationThread(real_time_factor=1.0)
    t.start()

    while True:
        color, depth = digits.render()
        digits.updateGUI(color, depth)

    t.stop()


if __name__ == "__main__":
    main()
