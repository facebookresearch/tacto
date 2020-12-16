# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2

import time
import pybullet as p
import pybulletX as px

import tacto


def render(sensor, n_times=1000):
    t = time.time()
    for i in range(n_times):
        sensor.update()
        sensor.render()

    elapsed = time.time() - t
    avg_time = elapsed / n_times
    print(f"Average time: {avg_time * 1000:.4f} ms [~ {1 / avg_time:.1f} fps]")


def test_rendering_fps(benchmark):
    sensor = tacto.Sensor(width=120, height=160, visualize_gui=True)

    px.init_pybullet(mode=p.DIRECT)

    digit = px.Body(
        "../meshes/digit.urdf", base_orientation=[0.0, -0.707106, 0.0, 0.707106],
    )

    obj = px.Body(
        "../examples/objects/sphere_small.urdf",
        base_position=[-0.015, 0, 0.035],
        global_scaling=0.15,
    )

    sensor.add_camera(digit.id, [-1])
    sensor.add_body(obj)

    # step the simulation till the ball falls on the digit camera
    for i in range(10):
        p.stepSimulation()

    # Use pytest-benchmark to benchmark the rendering performance
    # run 1000 times, the reported number will be millisecond instead of seconds
    benchmark(render, sensor)

    # visualize the resulting image to make sure it's not empty
    colors, depths = sensor.render()
    return colors[0]


if __name__ == "__main__":
    color = test_rendering_fps(lambda f, args: f(args, 1000))
    cv2.imwrite("color.jpg", color)
