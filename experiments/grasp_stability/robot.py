# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pybullet as pb


class Robot:
    def __init__(self, robotID):
        self.robotID = robotID

        # Get link/joint ID for arm
        self.armNames = [
            "right_j0",
            "right_j1",
            "right_j2",
            "right_j3",
            "right_j4",
            "right_j5",
            "right_j6",
        ]
        self.armJoints = self.get_id_by_name(self.armNames)
        self.armControlID = self.get_control_id_by_name(self.armNames)

        # Get link/joint ID for gripper
        self.gripperNames = [
            "base_joint_gripper_left",
            "base_joint_gripper_right",
        ]
        self.gripperJoints = self.get_id_by_name(self.gripperNames)
        self.gripperControlID = self.get_control_id_by_name(self.gripperNames)

        # Get ID for end effector
        self.eeName = ["right_hand"]
        self.eefID = self.get_id_by_name(self.eeName)[0]

        self.armHome = [
            -0.01863332,
            -1.30851021,
            -0.55159919,
            1.58025131,
            0.14144625,
            1.33963365,
            -1.98302146,
        ]
        self.pos = [0.53, 0, 0.215]
        # self.ori = [0, 1.5708, 1.5708]
        self.ori = [0, 3.14, np.pi / 2]
        self.width = 0.11

        self.tol = 1e-9

        self.init_robot()

    def get_id_by_name(self, names):
        """
        get joint/link ID by name
        """
        nbJoint = pb.getNumJoints(self.robotID)
        jointNames = {}
        for i in range(nbJoint):
            name = pb.getJointInfo(self.robotID, i)[1].decode()
            jointNames[name] = i

        return [jointNames[name] for name in names]

    def get_control_id_by_name(self, names):
        """
        get joint/link ID by name
        """
        nbJoint = pb.getNumJoints(self.robotID)
        jointNames = {}
        ctlID = 0
        for i in range(nbJoint):
            jointInfo = pb.getJointInfo(self.robotID, i)

            name = jointInfo[1].decode("utf-8")

            # skip fixed joint
            if jointInfo[2] == 4:
                continue

            # skip base joint
            if jointInfo[-1] == -1:
                continue

            jointNames[name] = ctlID
            ctlID += 1

        return [jointNames[name] for name in names]

    def reset_robot(self):
        for j in range(len(self.armJoints)):
            pb.resetJointState(self.robotID, self.armJoints[j], self.armHome[j])

    def init_robot(self):
        self.reset_robot()

        self.xin = pb.addUserDebugParameter("x", 0.3, 0.85, self.pos[0])
        self.yin = pb.addUserDebugParameter("y", -0.85, 0.85, self.pos[1])
        self.zin = pb.addUserDebugParameter("z", 0.0, 0.8, self.pos[2])
        self.widthin = pb.addUserDebugParameter("width", 0.03, 0.11, self.width)

        self.go(self.pos, self.ori, self.width)

    # Get the position and orientation of the UR5 end-effector
    def get_ee_pose(self):
        res = pb.getLinkState(self.robotID, self.eefID)

        world_positions = res[0]
        world_orientations = res[1]

        return (world_positions, world_orientations)

    # Get the joint angles (6 ur5 joints)
    def get_joint_angles(self):
        joint_angles = [_[0] for _ in pb.getJointStates(self.robotID, self.armJoints)]
        return joint_angles

    # Get the gripper width gripper width
    def get_gripper_width(self):
        width = 2 * np.abs(pb.getJointState(self.robotID, self.gripperJoints[-1])[0])
        return width

    # Go to the target pose
    def go(self, pos, ori=None, width=None, wait=False, gripForce=20):
        if ori is None:
            ori = self.ori

        if width is None:
            width = self.width

        ori_q = pb.getQuaternionFromEuler(ori)

        jointPose = pb.calculateInverseKinematics(self.robotID, self.eefID, pos, ori_q)
        jointPose = np.array(jointPose)

        jointPose[self.gripperControlID[0]] = -width / 2
        jointPose[self.gripperControlID[1]] = width / 2

        maxForces = np.ones(len(jointPose)) * 200
        maxForces[self.gripperControlID] = gripForce

        # Select the relavant joints for arm and gripper
        jointPose = jointPose[self.armControlID + self.gripperControlID]
        maxForces = maxForces[self.armControlID + self.gripperControlID]

        pb.setJointMotorControlArray(
            self.robotID,
            tuple(self.armJoints + self.gripperJoints),
            pb.POSITION_CONTROL,
            targetPositions=jointPose,
            forces=maxForces,
        )

        self.pos = pos
        if ori is not None:
            self.ori = ori
        if width is not None:
            self.width = width

        if wait:
            last_err = 1e6
            while True:
                pb.stepSimulation()
                ee_pose = self.get_ee_pose()
                w = self.get_gripper_width()
                err = (
                    np.sum(np.abs(np.array(ee_pose[0]) - pos))
                    + np.sum(np.abs(np.array(ee_pose[1]) - ori_q))
                    + np.abs(w - width)
                )
                diff_err = last_err - err
                last_err = err

                if np.abs(diff_err) < self.tol:
                    break

        # print("FINISHED")

    def update_input(self):

        x = pb.readUserDebugParameter(self.xin)
        y = pb.readUserDebugParameter(self.yin)
        z = pb.readUserDebugParameter(self.zin)
        width = pb.readUserDebugParameter(self.widthin)

        pos = [x, y, z]
        self.go(pos, width=width, wait=False)

    # Change the gripper to target width
    def gripper_control(self, width):
        pb.setJointMotorControlArray(
            self.robotID,
            self.gripperJoints,
            pb.POSITION_CONTROL,
            targetPositions=[-width / 2, width / 2],
        )
        self.width = width

    # Gripper close
    def gripper_close(self, width=None, max_torque=None):
        pass

    # Gripper open
    def gripper_open(self):
        self.gripper_control(0.11)
