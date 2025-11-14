# Copyright 2018 CNRS

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import numpy.linalg
from scipy.optimize import fmin_slsqp
from pinocchio import forwardKinematics, log, neutral, log6
import eigenpy

class CallbackLogger:
    def __init__(self):
        self.nfeval = 1
    def __call__(self,x):
        print('===CBK=== {0:4d}   {1}'.format(self.nfeval, x))
        self.nfeval += 1

def normalized_quaternion(q):
    return numpy.linalg.norm(q[3:7]) - 1

# This class computes inverse kinematics by numerical optimization
#
# The input to the inverse kinematic is provided by members
#  - leftFootRefPose
#  - rightFootRefPose
#  - waistRefPose
# that contain an element of type pinocchio SE3.
# Method solve computes a configuration in such a way that the poses of the
# end effector coincide with their reference poses.
class InverseKinematics (object):
    leftFootJoint = 'left_leg_6_joint'
    rightFootJoint = 'right_leg_6_joint'
    waistJoint = 'waist_joint'

    def __init__ (self, robot):
        self.robot = robot
        self.data = self.robot.model.createData()
        q = neutral (robot.model)
        self.q = q.copy()
        self.fullConfigSize = len(q)
        forwardKinematics(robot.model, self.data, q)

        # Initialize references of feet and center of mass with initial values
        # this will be the target to reach
        self.leftFootRefPose    = self.data.oMi [robot.leftFootJointId].copy ()
        self.rightFootRefPose   = self.data.oMi [robot.rightFootJointId].copy ()
        self.waistRefPose       = self.data.oMi [robot.waistJointId].copy ()

    def cost (self, q):
        forwardKinematics(self.robot.model, self.data, q)
        leftFootPose  = self.data.oMi [self.robot.leftFootJointId].copy()
        rightFootPose = self.data.oMi [self.robot.rightFootJointId].copy()
        waistPose     = self.data.oMi [self.robot.waistJointId].copy()

        logLeftFoot  = log6(self.leftFootRefPose.inverse() * leftFootPose)
        logRightFoot = log6(self.rightFootRefPose.inverse() * rightFootPose)
        logWaist     = log6(self.waistRefPose.inverse() * waistPose)

        return np.dot(logLeftFoot, logLeftFoot) + np.dot(logRightFoot, logRightFoot) + np.dot(logWaist, logWaist)

    def solve (self, q):
        return fmin_slsqp(self.cost,q,
                    f_eqcons=normalized_quaternion,
                    iprint=2, full_output=1)[0]


if __name__ == "__main__":
    from talos import Robot
    from pinocchio import neutral
    import numpy as np
    from inverse_kinematics import InverseKinematics
    import eigenpy

    robot = Robot ()
    
    ik = InverseKinematics (robot)
    ik.rightFootRefPose.translation = np.array ([0, -0.1, 0.1])
    ik.leftFootRefPose.translation = np.array ([0, 0.1, 0.1])
    ik.waistRefPose.translation = np.array ([0, 0, 0.95])

    q0 = neutral (robot.model)
    robot.display(q0)
    q0 [robot.name_to_config_index["leg_right_4_joint"]] = .2
    q0 [robot.name_to_config_index["leg_left_4_joint"]] = .2
    q0 [robot.name_to_config_index["arm_left_2_joint"]] = .2
    q0 [robot.name_to_config_index["arm_right_2_joint"]] = -.2
    q = ik.solve (q0)
    print(q)
    robot.display(q)
