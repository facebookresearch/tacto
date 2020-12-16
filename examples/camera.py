# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pybullet as p


class Camera:
    def __init__(self, width=320, height=240):
        self.width = width
        self.height = height

        camTargetPos = [0.5, 0, 0.05]
        camDistance = 0.4
        upAxisIndex = 2

        yaw = 90
        pitch = -30.0
        roll = 0

        fov = 60
        nearPlane = 0.01
        farPlane = 100

        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(
            camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex
        )

        aspect = width / height

        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov, aspect, nearPlane, farPlane
        )

    def get_image(self):
        img_arr = p.getCameraImage(
            self.width,
            self.height,
            self.viewMatrix,
            self.projectionMatrix,
            shadow=1,
            lightDirection=[1, 1, 1],
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb = img_arr[2]  # color image RGB H x W x 3 (uint8)
        dep = img_arr[3]  # depth image H x W (float32)
        return rgb, dep
