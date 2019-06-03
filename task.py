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
        mission_failed = -10 * self.get_distance() if self.sim.time > self.sim.runtime else 0
        target_reached_bonus = 5 if self.reached_target() else 0
        reward = np.clip(3. - 0.25 * (self.get_distance()), -1, 1)
        reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        reward_z = self.get_velocity_discounts()[2] * self.get_distance_rewards()[2]
        torque_penalty = 1. - np.clip(sum(self.get_torques()), 0, 1)
        return reward# + target_reached_bonus - crash_penalty

    def reached_target(self, threshold=0.2):
        return self.get_distance() <= threshold

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds, done)
            pose_all.append(self.sim.pose)
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
        return [1. - d ** 0.4 for d in self.get_normalized_distances()]

    def get_velocity_discounts(self):
        return [(1. - max(min(velocity, 475.) / 475., 0.1)) ** (1 / max(dist, 0.1)) for velocity, dist in
                zip(self.sim.v, self.get_normalized_distances())]

    def get_crash_penalty(self, done, vel_threshold):
        for i in range(3):
            # if self.sim.pose[i] <= self.sim.lower_bounds[i] + 5. or self.sim.pose[i] > self.sim.upper_bounds[i] - 5.:
            #     if self.sim.v[i] >= vel_threshold:
            #         return -10  # about to crush
            if done and (self.sim.pose[i] <= self.sim.lower_bounds[i] or self.sim.pose[i] > self.sim.upper_bounds[i]):
                return 10  # already crushed
        return 0

    def get_torques(self):
        return [ang_accel * mom_of_inertia for ang_accel, mom_of_inertia in
                zip(self.sim.angular_accels, self.sim.moments_of_inertia)]


def distances(pos1, pos2):
    return [distance(p1, p2) for p1, p2 in zip(pos1, pos2)]


def distance_3d(pos1, pos2):
    return np.sqrt(
        np.power((pos1[0] - pos2[0]), 2) + np.power((pos1[1] - pos2[1]), 2) + np.power((pos1[2] - pos2[2]), 2))


def distance(p1, p2):
    return np.sqrt(np.power(p1 - p2, 2.))
