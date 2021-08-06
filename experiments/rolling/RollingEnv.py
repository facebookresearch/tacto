# # Copyright (c) Facebook, Inc. and its affiliates.

# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.

import time

import cv2
import deepdish as dd
import numpy as np
import pybullet as pb
import pybullet_data
import tacto


class Camera:
    def __init__(self, cameraResolution=[320, 240]):
        self.cameraResolution = cameraResolution

        camTargetPos = [-0.01, 0, 0.04]
        camDistance = 0.05
        upAxisIndex = 2

        yaw = 0
        pitch = -20.0
        roll = 0

        fov = 60
        nearPlane = 0.01
        farPlane = 100

        self.viewMatrix = pb.computeViewMatrixFromYawPitchRoll(
            camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex
        )

        aspect = cameraResolution[0] / cameraResolution[1]

        self.projectionMatrix = pb.computeProjectionMatrixFOV(
            fov, aspect, nearPlane, farPlane
        )

    def get_image(self):
        img_arr = pb.getCameraImage(
            self.cameraResolution[0],
            self.cameraResolution[1],
            self.viewMatrix,
            self.projectionMatrix,
            shadow=1,
            lightDirection=[1, 1, 1],
            renderer=pb.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb = img_arr[2]  # color data RGB
        dep = img_arr[3]  # depth data
        return rgb, dep


def draw_circle(img, state):
    if state is None:
        return img

    # Center coordinates
    center_coordinates = (int(state[1] * img.shape[1]), int(state[0] * img.shape[0]))

    # Radius of circle
    radius = 7

    # Red color in BGR
    color = (255, 255, 255)

    # Line thickness of -1 px
    thickness = -1

    # Using cv2.circle() method
    # Draw a circle of red color of thickness -1 px
    img = cv2.circle(img, center_coordinates, radius, color, thickness)

    return img


class RollingEnv:
    def __init__(
        self,
        tactoResolution=[120, 160],
        visPyBullet=True,
        visTacto=True,
        recordLogs=False,
        skipFrame=1,
    ):
        """
        Initialize

        Args:
            tactoResolution: tactile output resolution, Default: [120, 160]
            visPyBullet: whether display pybullet, Default: True
            visTacto: whether display tacto GUI, Default: True
            skipFrame: execute the same action for skipFrame+1 frames.
                       Save time to perform longer horizon
        """
        self.tactoResolution = tactoResolution
        self.visPyBullet = visPyBullet
        self.visTacto = visTacto
        self.skipFrame = skipFrame

        self.error = 0

        self.create_scene()

        self.cam = Camera(cameraResolution=[320, 240])
        self.logs = {"touch": [], "vision": [], "states": [], "goal": None}
        self.recordLogs = recordLogs

    def create_scene(self):
        """
        Create scene and tacto simulator
        """

        # Initialize digits
        digits = tacto.Sensor(
            width=self.tactoResolution[0],
            height=self.tactoResolution[1],
            visualize_gui=self.visTacto,
        )

        if self.visPyBullet:
            self.physicsClient = pb.connect(pb.GUI)
        else:
            self.physicsClient = pb.connect(pb.DIRECT)

        pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        pb.setGravity(0, 0, -9.81)  # Major Tom to planet Earth

        # Set camera view
        pb.resetDebugVisualizerCamera(
            cameraDistance=0.12,
            cameraYaw=0,
            cameraPitch=-20,
            cameraTargetPosition=[0, 0, 0.02],
        )

        pb.loadURDF("plane.urdf")  # Create plane

        digitURDF = "setup/sensors/digit.urdf"
        # Set upper digit
        digitPos1 = [0, 0, 0.011]
        digitOrn1 = pb.getQuaternionFromEuler([0, -np.pi / 2, 0])
        digitID1 = pb.loadURDF(
            digitURDF,
            basePosition=digitPos1,
            baseOrientation=digitOrn1,
            useFixedBase=True,
        )
        digits.add_camera(digitID1, [-1])

        # Set lower digit
        digitPos2 = [0, 0, 0.07]
        digitOrn2 = pb.getQuaternionFromEuler([0, np.pi / 2, np.pi])
        digitID2 = pb.loadURDF(
            digitURDF, basePosition=digitPos2, baseOrientation=digitOrn2,
        )
        digits.add_camera(digitID2, [-1])

        # Create object and GUI controls
        init_xyz = np.array([0, 0.0, 8])

        # Add object to pybullet and tacto simulator
        urdfObj = "setup/objects/sphere_small.urdf"
        objPos = np.array([-1.5, 0, 4]) / 100
        objOrn = pb.getQuaternionFromEuler([0, 0, 0])
        globalScaling = 0.15

        # Add ball urdf into pybullet and tacto
        objId = digits.loadURDF(urdfObj, objPos, objOrn, globalScaling=globalScaling)

        # Add constraint to movable digit (upper)
        cid = pb.createConstraint(
            digitID2, -1, -1, -1, pb.JOINT_FIXED, [0, 0, 0], [0, 0, 0], init_xyz / 100
        )

        # Save variables
        self.digits = digits

        self.digitID1, self.digitPos1, self.digitOrn1 = digitID1, digitPos1, digitOrn1
        self.digitID2, self.digitPos2, self.digitOrn2 = digitID2, digitPos2, digitOrn2
        self.objId, self.objPos, self.objOrn = objId, objPos, objOrn
        self.cid = cid

    def reset(self):
        """
        Reset environment
        """

        pb.resetBasePositionAndOrientation(self.objId, self.objPos, self.objOrn)
        pb.resetBasePositionAndOrientation(
            self.digitID2, self.digitPos2, self.digitOrn2
        )

        xyz = [0, 0, 0.055]
        pb.changeConstraint(self.cid, xyz, maxForce=5)

        for i in range(10):
            pb.stepSimulation()
        self.digits.update()

        self.error = 0

        self.logs = {"touch": [], "vision": [], "states": [], "goal": None}

    def pose_estimation(self, color, depth):
        """
        Estimate location of the ball
        For simplicity, using depth to get the ball center. Can be replaced by more advanced perception system.
        """
        ind = np.unravel_index(np.argmax(depth, axis=None), depth.shape)
        maxDepth = depth[ind]

        if maxDepth < 0.0005:
            return None

        center = np.array(
            [ind[0] / self.tactoResolution[1], ind[1] / self.tactoResolution[0]]
        )

        return center

    def controller_Kx(self, state, goal, vel_last, K):
        """
        Args:
            state: current ball location, range 0-1, e.g. np.array([0.5, 0.5])
            goal: goal ball location, range 0-1, e.g. np.array([0.5, 0.5])
            vel_last: previous velocity
            K: control parameters

        vel = K*(goal-state)^T
        """
        if state is None:
            # return np.array([0.0, 0.0])
            return vel_last

        error = np.matrix(goal - state).T
        vel = K.dot(error)
        vel = np.array(vel).T[0]

        return vel

    def step(self, render=True):
        """
        Step simulation, sync with tacto simulator

        Args:
            render: whether render tactile imprints, Default: True
        """
        # Step in pybullet
        pb.stepSimulation()

        if not (render):
            return

        st = time.time()
        # Sync tacto
        self.digits.update()
        # Render tactile imprints
        self.color, self.depth = self.digits.render()

        self.time_render.append(time.time() - st)
        self.time_render = self.time_render[-100:]
        # print("render {:.4f}s".format(np.mean(self.time_render)), end=" ")
        st = time.time()

        if self.recordLogs:
            self.logs["touch"].append([self.color.copy(), self.depth])

            self.visionColor, self.visionDepth = self.cam.get_image()
            self.logs["vision"].append([self.visionColor, self.visionDepth])

        if self.visTacto:
            color1 = self.color[1].copy()
            x0 = int(self.goal[0] * self.tactoResolution[1])
            y0 = int(self.goal[1] * self.tactoResolution[0])
            color1[x0 - 4 : x0 + 4, y0 - 4 : y0 + 4, :] = [255, 255, 255]

            # color1 = draw_circle(color1, self.goal)
            self.color[1] = color1
            self.digits.updateGUI(self.color, self.depth)

        self.time_vis.append(time.time() - st)
        self.time_vis = self.time_vis[-100:]

        # print("visualize {:.4f}s".format(np.mean(self.time_vis)))

    def save_logs(self, fn):
        dd.io.save(fn, self.logs)

    def cost_function(self, state, goal, vel, xyz=[0, 0, 0]):
        if state is None:
            distance = 1
        else:
            distance = np.sum((state - goal) ** 2) ** 0.5

        cost_dist = distance

        return cost_dist

    def simulate(self, goal, K):
        """
        Simulate rolling ball to goal location with PD control, based on tactile signal
            goal: goal ball location in tactile space, range 0-1, e.g. np.array([0.5, 0.5])
            K: vel = K.dot(goal-state), e.g. np.array([[1,2],[3,4]])
        """

        # Main simulation loop
        simulation_time = 31  # Duration simulation
        self.goal = goal

        self.time_render = []
        self.time_vis = []

        self.reset()
        xyz = [0, 0, 0.055]
        costs = 0.0
        vel = [0, 0]

        self.logs["goal"] = goal

        for i in range(simulation_time):
            pb.changeConstraint(self.cid, xyz, maxForce=5)

            # position, orientation = self.digits.get_pose(self.objId, -1)

            self.step()

            state = self.pose_estimation(self.color[1], self.depth[1])
            vel = self.controller_Kx(state, goal, vel, K)

            self.logs["states"].append(state)

            xyz[:2] += vel

            for _ in range(self.skipFrame):
                xyz[:2] += vel
                self.step(render=False)

                c = self.cost_function(state, goal, vel, xyz=xyz)
                costs += c

            c = self.cost_function(state, goal, vel, xyz=xyz)
            costs += c

        return costs


if __name__ == "__main__":
    # env = RollingEnv(recordLogs=True, skipFrame=1, tactoResolution=[240, 320])
    env = RollingEnv(skipFrame=2)

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

    # # iter 5
    # K = np.array([-0.2931, 1.4576, -1.7034, -1.8010]) / 1000

    # iter 30
    K = np.array([-2.0000, 0.3131, -0.1268, -2.0000]) / 1000

    for goal in goals:
        rewards = env.simulate(goal, K.reshape(2, 2))
