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

import time
import numpy as np
from math import atan2, cos, sin
from pinocchio import centerOfMass, forwardKinematics
from cop_des import CoPDes
from com_trajectory import ComTrajectory
from inverse_kinematics import InverseKinematics
from tools import Constant, Piecewise, Affine

# Computes the trajectory of a swing foot.
#
# Input data are
#  - initial and final time of the trajectory,
#  - initial and final pose of the foot,
#  - maximal height of the foot,
#
# The trajectory is polynomial with zero velocities at start and end.
# The orientation of the foot is kept as in intial pose.
class SwingFootTrajectory(object):
    def __init__(self, t_init, t_end, init, end, height):
        assert(init[2] == end[2])
        self.t_init = t_init
        self.t_end = t_end
        self.height = height
        self.init = init
        self.end = end
        self.delta_t = t_end - t_init

        self.foot_trajectory = None
        

    def __call__(self, t):
        
        if t < self.t_init:
            t = self.t_init
        if t > self.t_end:
            t = self.t_end

        s = (t - self.t_init) / self.delta_t
        s2 = s * s
        s3 = s2 * s

        # 3rd order polynomial for x and y
        x = self.init[0] + (self.end[0] - self.init[0]) * (3 * s2 - 2 * s3)
        y = self.init[1] + (self.end[1] - self.init[1]) * (3 * s2 - 2 * s3)

        # 4th order polynomial for z
        z = self.init[2] + self.height * 16 * s2 * (s - 1)**2

        return np.array([x, y, z])

# Computes a walking whole-body motion
#
# Input data are
#  - an initial configuration of the robot,
#  - a sequence of step positions (x,y,theta) on the ground,
#  - a mapping from time to R corresponding to the desired orientation of the
#    waist. If not provided, keep constant orientation.
#
class WalkingMotion(object):
    step_height = 0.05

    def __init__(self, robot):
        self.robot = robot
        self.ik = InverseKinematics (robot)
        self.times = np.arange(0, 20, 0.01)

    def compute(self, q0, steps, waistOrientation = None):
        # Test input data
        if len(steps) < 4:
            raise RuntimeError("sequence of step should be of length at least 4 instead of " +
                               f"{len(steps)}")
        # Copy steps in order to avoid modifying the input list.
        steps_ = steps[:]
        # Compute offset between waist and center of mass since we control the center of mass
        # indirectly by controlling the waist.
        data = self.robot.model.createData()
        #forwardKinematics(self.robot.model, data, q0)
        com = centerOfMass(self.robot.model, data, q0)
        waist_pose = data.oMi[self.robot.waistJointId]
        com_offset = waist_pose.translation - com
        # Trajectory of left and right feet
        self.lf_traj = Piecewise()
        self.rf_traj = Piecewise()
        # write your code here

        t_ds = CoPDes.double_support_time
        t_ss = CoPDes.single_support_time

        # Initial foot positions
        self.rf_traj.segments.append(Constant(0., t_ds, steps_[0]))
        self.lf_traj.segments.append(Constant(0., t_ds, steps_[1]))

        # Steps
        for i in range(1, (len(steps_) - 2)//2 + 1):
            self.rf_traj.segments.append(SwingFootTrajectory(t_ds + (i-1)*(t_ss + t_ds)*2,
                                                            t_ds + (i-1)*(t_ss + t_ds)*2 + t_ss,
                                                            steps_[2*i - 2], steps_[2*i], WalkingMotion.step_height))
            self.lf_traj.segments.append(Constant(t_ds + (i-1)*(t_ss + t_ds)*2,
                                                  t_ds + (i-1)*(t_ss + t_ds)*2 + t_ss,
                                                  steps_[2*i - 1]))

            self.rf_traj.segments.append(Constant(t_ds + (i-1)*(t_ss + t_ds)*2 + t_ss,
                                                  t_ds + i*(t_ss + t_ds)*2,
                                                  steps_[2*i]))
            self.lf_traj.segments.append(SwingFootTrajectory(t_ds + (i-1)*(t_ss + t_ds)*2 + t_ss,
                                                            t_ds + i*(t_ss + t_ds)*2,
                                                            steps_[2*i - 1], steps_[2*i + 1], WalkingMotion.step_height))

        # Final foot positions
        n = (len(steps_) - 2)//2
        self.rf_traj.segments.append(Constant(t_ds + n*(t_ss + t_ds)*2,
                                              t_ds + n*(t_ss + t_ds)*2 + t_ss,
                                              steps_[-2]))
        self.lf_traj.segments.append(Constant(t_ds + n*(t_ss + t_ds)*2,
                                              t_ds + n*(t_ss + t_ds)*2 + t_ss,
                                              steps_[-1]))
        
        ##Plot foot trajectories
        #import matplotlib.pyplot as plt
        #times = 0.01 * np.arange(int((self.rf_traj.segments[-1].t_end)/0.01)+1)
        #rf = np.array(list(map(self.rf_traj, times)))
        #lf = np.array(list(map(self.lf_traj, times)))
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #ax.set_xlabel("second")
        #ax.set_ylabel("meter")
        #ax.plot(times, lf[:,0], label="x left foot")
        #ax.plot(times, rf[:,0], label="x right foot")
        #ax.plot(times, lf[:,1], label="y left foot")
        #ax.plot(times, rf[:,1], label="y right foot")
        #ax.plot(times, lf[:,2], label="z left foot")
        #ax.plot(times, rf[:,2], label="z right foot")
        #ax.legend()
        #plt.show()

        # Compute trajectory of the center of mass
        start = steps_[0][:2]           # first foot step position
        end = steps_[-1][:2]            # last foot step position
        foot_positions = [s[:2] for s in steps_]  # only x,y of steps
        com_traj = ComTrajectory(start, foot_positions, end, 0.95)
        self.com_trajectory = com_traj
        com_traj.compute()

        configs = []
        
        for i in range(len(self.times)):
            t = self.times[i]
            # Compute desired positions of left and right foot
            left_foot_pos = self.lf_traj(t)
            right_foot_pos = self.rf_traj(t)

            # Set desired foot positions
            self.ik.leftFootRefPose.translation = left_foot_pos
            self.ik.rightFootRefPose.translation = right_foot_pos
            
            # Set desired waist position
            com_pos = com_traj(t)
            waist_pos = com_pos + com_offset
            self.ik.waistRefPose.translation = waist_pos

            # Solve inverse kinematics
            if i == 0:
                q_init = q0
            else:
                q_init = configs[-1]
            q_sol = self.ik.solve(q_init)
            configs.append(q_sol)

        return configs

def main():
    import matplotlib.pyplot as plt
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
    q0 [robot.name_to_config_index["leg_right_4_joint"]] = .2
    q0 [robot.name_to_config_index["leg_left_4_joint"]] = .2
    q0 [robot.name_to_config_index["arm_left_2_joint"]] = .2
    q0 [robot.name_to_config_index["arm_right_2_joint"]] = -.2
    q = ik.solve (q0)
    robot.display(q)


    ## Test SwalkingMotion
    ##sft = SwingFootTrajectory(0., 2., np.array([0., 0., 0.]), np.array([0.2, 0., 0.]), 0.05)
    ##times = 0.01 * np.arange(200)
    ##foot = np.array(list(map(sft, times)))
    ##fig = plt.figure()
    ##ax = fig.add_subplot(111)
    ##ax.set_xlabel("second")
    ##ax.set_ylabel("meter")
    ##ax.plot(times, foot[:,0], label="x_foot")
    ##ax.plot(times, foot[:,1], label="y_foot")
    ##ax.plot(times, foot[:,2], label="z_foot")
    ##ax.legend()
    ##plt.show()

    wm = WalkingMotion(robot)
    # First two values correspond to initial position of feet
    # Last two values correspond to final position of feet
    steps = [np.array([0.0, -.1, 0.1]), np.array([0.0, .1, 0.1]), 
             np.array([0.4, -.1, 0.1]),np.array([.8, .1, 0.1]), 
             np.array([1.2, -.1, 0.1]),np.array([1.6, .1, 0.1]), 
             np.array([1.6, -.1, 0.1]), np.array([1.6, .1, 0.1])]
    configs = wm.compute(q, steps)
    #print(len(configs))
    for q in configs:
        time.sleep(2e-2)
        robot.display(q)
    delta_t = wm.com_trajectory.delta_t
    times = delta_t*np.arange(wm.com_trajectory.N+1)
    lf = np.array(list(map(wm.lf_traj, times)))
    rf = np.array(list(map(wm.rf_traj, times)))
    cop_des = np.array(list(map(wm.com_trajectory.cop_des, times)))
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.plot(times, lf[:,0], label="x left foot")
    ax1.plot(times, rf[:,0], label="x right foot")
    ax1.plot(times, cop_des[:,0], label="x CoPdes")
    ax1.legend()
    ax2.plot(times, lf[:,1], label="y left foot")
    ax2.plot(times, rf[:,1], label="y right foot")
    ax2.plot(times, cop_des[:,1], label="y CoPdes")
    ax2.legend()
    ax3.plot(times, lf[:,2], label="z left foot")
    ax3.plot(times, rf[:,2], label="z right foot")
    ax3.legend()
    plt.show()


if __name__ == "__main__":
    main()

