# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import logging
import collections

import cv2
import gym
from gym.envs.registration import register

import numpy as np
import pybullet as p
from attrdict import AttrMap
from omegaconf import OmegaConf

import pybulletX as px

import tacto
from sawyer_gripper import SawyerGripper
from camera import Camera

_log = logging.getLogger(__name__)

def _get_dtype_min_max(dtype):
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).min, np.iinfo(dtype).max
    if np.issubdtype(dtype, np.floating):
        return np.finfo(dtype).min, np.finfo(dtype).max
    raise NotImplementedError


def convert_obs_to_obs_space(obs):
    if isinstance(obs, (int, float)):
        return convert_obs_to_obs_space(np.array(obs))

    if isinstance(obs, np.ndarray):
        min_, max_ = _get_dtype_min_max(obs.dtype)
        return gym.spaces.Box(low=min_, high=max_, shape=obs.shape, dtype=obs.dtype)

    # for list-like container
    if isinstance(obs, list) or isinstance(obs, tuple):
        if np.all([isinstance(_, float) for _ in obs]):
            return convert_obs_to_obs_space(np.array(obs))
        return gym.spaces.Tuple([convert_obs_to_obs_space(_) for _ in obs])

    # for any dict-like container
    if isinstance(obs, collections.abc.Mapping):
        # SpaceDict inherits from gym.spaces.Dict and provides more functionalities
        return px.utils.SpaceDict({k: convert_obs_to_obs_space(v) for k, v in obs.items()})

def _get_default_config_path():
    filename = "conf/sawyer_gripper_env.yaml"
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)


class SawyerGripperEnv(gym.Env):
    reward_per_step = -0.01

    def __init__(self, config_path=_get_default_config_path()):
        px.init(mode=p.GUI)
        self.cfg = OmegaConf.load(config_path)
        self.robot = SawyerGripper(**self.cfg.sawyer_gripper)
        self.obj = px.Body(**self.cfg.object)
        self.digits = tacto.Sensor(**self.cfg.tacto)
        self.camera = Camera()
        self.viewer = None

        self.digits.add_camera(self.robot.id, self.robot.digit_links)

        self.digits.add_body(self.obj)

        self.reset()

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended. Further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        done = self._done()
        reward = self.reward_per_step + int(done)
        info = {}

        self.robot.set_actions(action)

        p.stepSimulation()

        self.obs = self._get_obs()

        return self.obs, reward, done, info

    def _done(self):
        (x, y, z), _ = self.obj.get_base_pose()
        velocity, angular_velocity = self.obj.get_base_velocity()
        velocity = np.linalg.norm(velocity)
        angular_velocity = np.linalg.norm(angular_velocity)

        _log.debug(
            f"obj.z: {z}, obj.velocity: {velocity:.4f}, obj.angular_velocity: {angular_velocity:.4f}"
        )
        return z > 0.1 and velocity < 0.025 and angular_velocity < 0.025

    def _get_obs(self):
        cam_color, cam_depth = self.camera.get_image()

        # update objects positions registered with digits
        self.digits.update()
        colors, depths = self.digits.render()

        obj_pose = self.obj.get_base_pose()

        return AttrMap(
            {
                "camera": {"color": cam_color, "depth": cam_depth},
                "digits": [
                    {"color": color, "depth": depth}
                    for color, depth in zip(colors, depths)
                ],
                "robot": self.robot.get_states(),
                "object": {
                    "position": np.array(obj_pose[0]),
                    "orientation": np.array(obj_pose[1]),
                },
            }
        )

    def reset(self):
        self.robot.reset()

        # Move the object to random location
        dx, dy = np.random.randn(2) * 0.1
        x, y, z = self.obj.init_base_position
        position = [x + dx, y + dy, z]
        self.obj.set_base_pose(position)

        # get initial observation
        self.obs = self._get_obs()

        return self.obs

    def render(self, mode="human"):
        def _to_uint8(depth):
            min_, max_ = depth.min(), depth.max()
            return ((depth - min_) / (max_ - min_) * 255).astype(np.uint8)

        img = np.concatenate([digit.color for digit in self.obs.digits], axis=1)
        cv2.imshow("img", img)
        cv2.waitKey(1)

    def close(self):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)

    @property
    def observation_space(self):
        """
        >>> print(self.observation_space)
        Dict(
            camera: Dict(color:Box(0, 255, (240, 320, 4), uint8),
            depth: Box(-3.402823e+38, 3.402823e+38, (240, 320), float32)),
            digits: Tuple(
                Dict(
                    color: Box(0, 255, (160, 120, 3), uint8),
                    depth: Box(-3.402823+38, 3.402823e+38, (160, 120), float32)
                ),
                Dict(
                    color: Box(0, 255, (160, 120, 3), uint8),
                    depth: Box(-3.402823+38, 3.402823e+38, (160, 120), float32)
                )
            ),
            end_effector: Dict(
                orientation: Box(-3.1415927, 3.1415927, (4,), float32),
                position: Box(-0.85, 0.85, (3,), float32)
            ),
            gripper_width: Box(0.03, 0.11, (1,), float32)
        )
        """
        return px.utils.SpaceDict(
            {
                "camera": convert_obs_to_obs_space(self.obs.camera),
                "digits": convert_obs_to_obs_space(self.obs.digits),
                "robot": self.robot.state_space,
                "object": {
                    "position": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                    "orientation": gym.spaces.Box(low=-1.0, high=+1.0, shape=(4,)),
                },
            }
        )

    @property
    def action_space(self):
        action_space = copy.deepcopy(self.robot.action_space)
        del action_space["wait"]
        return action_space

def make_sawyer_gripper_env():
    env = SawyerGripperEnv()
    return env

register(
    id="sawyer-gripper-v0", entry_point="sawyer_gripper_env:make_sawyer_gripper_env",
)
