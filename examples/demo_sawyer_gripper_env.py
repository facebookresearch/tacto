# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import gym
import sawyer_gripper_env  # noqa: F401


class GraspingPolicy(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.t = 0

    def forward(self, states=None):
        action = self.env.action_space.new()
        if not states:
            return action

        z_low, z_high = 0.05, 0.4
        dz = 0.02
        w_open, w_close = 0.11, 0.05
        gripper_force = 20

        if self.t < 50:
            action.end_effector.position = states.object.position + [0, 0, z_high]
            action.end_effector.orientation = [0.0, 1, 0.0, 0.0]
            action.gripper_width = w_open
        elif self.t < 100:
            s = (self.t - 50) / 50
            z = z_high - s * (z_high - z_low)
            action.end_effector.position = states.object.position + [0, 0, z]
        elif self.t < 150:
            action.gripper_width = w_close
            action.gripper_force = gripper_force
        elif self.t < 220:
            delta = [0, 0, dz]
            action.end_effector.position = states.robot.end_effector.position + delta
            action.gripper_width = w_close
            action.gripper_force = gripper_force
        else:
            action.gripper_width = w_close

        self.t += 1

        return action


def main():
    env = gym.make("sawyer-gripper-v0")
    print (f"Env observation space: {env.observation_space}")
    env.reset()

    # Create a hard-coded grasping policy
    policy = GraspingPolicy(env)

    # Set the initial state (obs) to None, done to False
    obs, done = None, False

    while not done:
        env.render()
        action = policy(obs)
        obs, reward, done, info = env.step(action)

    env.close()


if __name__ == "__main__":
    main()
