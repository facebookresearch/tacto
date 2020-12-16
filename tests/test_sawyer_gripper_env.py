# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from tacto_exp.sawyer_gripper_env import SawyerGripperEnv


@pytest.mark.skip(reason="no way of currently testing this headlessly")
def test_sawyer_gripper_env():
    # TODO(poweic): make this `env = gym.make('CartPole-v0')`
    env = SawyerGripperEnv()

    obs_space = env.observation_space
    obs_space.sample()

    env.reset()
    for _ in range(2):
        env.render()

        # take a random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        print(obs.camera.color.shape, obs.camera.depth.shape)
        for digit in obs.digits:
            print(digit.color.shape, digit.depth.shape)

    env.close()
