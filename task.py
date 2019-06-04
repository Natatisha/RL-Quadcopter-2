import math

import numpy as np
from physics_sim import PhysicsSim


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self, rotor_speeds, done):
        """Uses current pose of sim to return reward."""
        crash_penalty = self.get_crash_penalty(done, 50.)
        reached_target = self.reached_target()
        mission_failed_penalty = 100. if self.sim.time > self.sim.runtime and not reached_target else 0
        target_reached_bonus = 100. if reached_target else 0
        # print("target_reached_bonus = ", target_reached_bonus)
        reward = np.clip(3. - 0.25 * (self.get_distance()), -1, 1)

        # normalize by max distance which is from 0 to target
        reward_z = 1. - (np.abs(self.sim.pose[2] - self.target_pos[2]) /
                         np.abs(self.target_pos[2] - self.sim.lower_bounds[2])) ** 0.4
        # print("dist reward = ", reward_z)

        # max pose is initial
        reward_xy = np.clip(1. - 0.25 * (eucl_distance(self.target_pos[:2], self.sim.pose[:2]) /
                                         max(0.001, eucl_distance(self.target_pos[:2], self.sim.init_pose[:2]))), -1, 1)
        # print(penalty)
        # torque_penalty = 1. - np.clip(sum(self.get_torques()), 0, 1)
        # print("torque ", torque_penalty)
        # print(" position {} \ntorques {} ".format(self.sim.pose, self.get_torques()))

        # solution that once shot is initial z 5 target 10 reward for z distance * penalty for angles
        weights = np.array([1., 1., 10.])  # make accent on z position but don't forget about x and y
        rewards_weighted = np.dot(self.get_distance_rewards(), weights) / sum(weights)  # normalize to range [0, 1]

        penalty = (1. - abs(math.sin(self.sim.pose[3])))
        penalty *= (1. - abs(math.sin(self.sim.pose[4])))
        penalty *= (1. - abs(math.sin(self.sim.pose[5])))
        # print("reward xy {} \npenalty {} \nreward_z {}".format(reward_xy, penalty, reward_z))
        # print("reward {} position {}".format(reward, self.sim.pose[:3]))
        return reward_z * penalty + target_reached_bonus - crash_penalty  # - mission_failed_penalty

    def reached_target(self, threshold=5.):
        # on target height or not higher than threshold
        return self.sim.pose[2] >= self.target_pos[2] and self.sim.pose[2] - self.target_pos[2] <= threshold

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds, done)
            pose_all.append(self.sim.pose)
            if self.reached_target():
                done = True
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state

    def get_distance(self):
        return distance_3d(self.sim.pose, self.target_pos)

    def get_normalized_distances(self):
        return [d / max_d for d, max_d in
                zip(distances(self.sim.pose, self.target_pos), distances(self.sim.upper_bounds, self.sim.lower_bounds))]

    def get_distance_rewards(self):
        return np.array([1. - d ** 0.4 for d in self.get_normalized_distances()])

    def get_velocity_discounts(self):
        return [(1. - max(min(velocity, 475.) / 475., 0.1)) ** (1 / max(dist, 0.1)) for velocity, dist in
                zip(self.sim.v, self.get_normalized_distances())]

    def get_crash_penalty(self, done, vel_threshold):
        for i in range(3):
            # if self.sim.pose[i] <= self.sim.lower_bounds[i] + 5. or self.sim.pose[i] > self.sim.upper_bounds[i] - 5.:
            #     if self.sim.v[i] >= vel_threshold:
            #         return -10  # about to crush
            if self.sim.pose[i] <= self.sim.lower_bounds[i] or self.sim.pose[i] > self.sim.upper_bounds[i]:
                return 15.  # already crushed
        return 0.

    def get_torques(self):
        return [ang_accel * mom_of_inertia for ang_accel, mom_of_inertia in
                zip(self.sim.angular_accels, self.sim.moments_of_inertia)]


def distances(pos1, pos2):
    return [distance(p1, p2) for p1, p2 in zip(pos1, pos2)]


def distance_3d(pos1, pos2):
    return np.sqrt(
        np.power((pos1[0] - pos2[0]), 2) + np.power((pos1[1] - pos2[1]), 2) + np.power((pos1[2] - pos2[2]), 2))


def eucl_distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def distance(p1, p2):
    return np.sqrt(np.power(p1 - p2, 2.))
