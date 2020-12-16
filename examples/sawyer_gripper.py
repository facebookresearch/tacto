# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import collections
import functools

import numpy as np
import pybullet as p

from gym import spaces

import pybulletX as px
from pybulletX.utils.space_dict import SpaceDict

log = logging.getLogger(__name__)


def _overwrite(d1, d2):
    # Given two dictionaries d1 and d2 (have the same structure), overwrite
    # value in d1 with value in d2 (if not None). Return a copy
    if isinstance(d1, collections.abc.Mapping):
        return d1.__class__({k: _overwrite(d1[k], d2.get(k)) for k in d1.keys()})

    if d2 is not None:
        return d2

    return d1


def _vectorize(s):
    return np.r_[s.end_effector.position, s.end_effector.orientation, s.gripper_width]


class SawyerGripper(px.Robot):
    end_effector_name = "right_hand"
    gripper_names = [
        "base_joint_gripper_left",
        "base_joint_gripper_right",
    ]
    digit_joint_names = ["joint_finger_tip_left", "joint_finger_tip_right"]

    MAX_FORCES = 200

    def __init__(self, robot_params, init_state):
        super().__init__(**robot_params)

        self.zero_pose = self._states_to_joint_position(init_state)
        self.reset()

    @property
    @functools.lru_cache(maxsize=None)
    def state_space(self):
        return SpaceDict(
            {
                "end_effector": {
                    "position": spaces.Box(
                        low=np.array([0.3, -0.85, 0]),
                        high=np.array([0.85, 0.85, 0.8]),
                        shape=(3,),
                        dtype=np.float32,
                    ),
                    "orientation": spaces.Box(
                        low=-np.pi, high=np.pi, shape=(4,), dtype=np.float32
                    ),
                },
                "gripper_width": spaces.Box(
                    low=0.03, high=0.11, shape=(1,), dtype=np.float32
                ),
            }
        )

    @property
    @functools.lru_cache(maxsize=None)
    def action_space(self):
        action_space = copy.deepcopy(self.state_space)
        action_space["gripper_force"] = spaces.Box(
            low=0, high=self.MAX_FORCES, shape=(1,)
        )
        action_space["wait"] = spaces.MultiBinary(n=1)
        return action_space

    def get_states(self):
        ee_link = self.get_link_state_by_name(self.end_effector_name)
        gripper_joint = self.get_joint_states()[-1]

        states = self.state_space.new()
        states.end_effector.position = np.array(ee_link.link_world_position)
        states.end_effector.orientation = np.array(ee_link.link_world_orientation)
        states.gripper_width = 2 * np.abs(gripper_joint.joint_position)
        return states

    def _states_to_joint_position(self, states):
        eef_id = self.get_joint_index_by_name(self.end_effector_name)

        joint_position = np.array(
            p.calculateInverseKinematics(
                self.id,
                eef_id,
                states.end_effector.position,
                states.end_effector.orientation,
                maxNumIterations=100,
                residualThreshold=1e-5,
            )
        )

        joint_position[self.gripper_joint_ids[0]] = -states.gripper_width / 2
        joint_position[self.gripper_joint_ids[1]] = states.gripper_width / 2
        return joint_position

    def set_actions(self, actions):
        states = self.get_states()

        # action is the desired state, overwrite states with actions to get it
        desired_states = _overwrite(states, actions)

        joint_position = self._states_to_joint_position(desired_states)

        max_forces = np.ones(self.num_dofs) * self.MAX_FORCES
        if actions.get("gripper_force"):
            max_forces[self.gripper_joint_ids] = actions["gripper_force"]

        self.set_joint_position(
            joint_position, max_forces, use_joint_effort_limits=False
        )

    @property
    def digit_links(self):
        return [self.get_joint_index_by_name(name) for name in self.digit_joint_names]

    def go(self, pos, ori=None, width=None, grip_force=20):
        action = self.action_space.new()
        action.end_effector.position = pos
        if ori:
            action.end_effector.orientation = p.getQuaternionFromEuler(ori)
        action.gripper_width = width
        action.gripper_force = grip_force
        self.set_actions(action)

    @property
    @functools.lru_cache(maxsize=None)
    def gripper_joint_ids(self):
        return [
            self.free_joint_indices.index(self.get_joint_index_by_name(name))
            for name in self.gripper_names
        ]
