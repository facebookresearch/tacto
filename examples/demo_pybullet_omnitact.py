# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import hydra
import pybullet as p
import tacto  # Import TACTO

import pybulletX as px

log = logging.getLogger(__name__)


# Load the config YAML file from examples/conf/omnitact.yaml
@hydra.main(config_path="conf", config_name="omnitact")
def main(cfg):
    # Initialize OmniTact
    omnitact = tacto.Sensor(
        **cfg.tacto, **{"config_path": tacto.get_omnitact_config_path()}
    )

    # Initialize World
    log.info("Initializing world")
    px.init()

    p.resetDebugVisualizerCamera(**cfg.pybullet_camera)
    p.configureDebugVisualizer(
        p.COV_ENABLE_SHADOWS, 1, lightPosition=[50, 0, 80],
    )

    # Create and initialize OmniTact
    body = px.Body(**cfg.omnitact)
    omnitact.add_camera(body.id, [-1])

    obj = px.Body(**cfg.object)
    omnitact.add_body(obj)

    # Create control panel to control the 6DoF pose of the object
    panel = px.gui.PoseControlPanel(obj, **cfg.object_control_panel)
    panel.start()
    log.info("Use the slides to move the object until in contact with the OmniTact")

    # run p.stepSimulation in another thread
    t = px.utils.SimulationThread(real_time_factor=1.0)
    t.start()

    while True:
        color, depth = omnitact.render()
        omnitact.updateGUI(color, depth)

    t.stop()


if __name__ == "__main__":
    main()
