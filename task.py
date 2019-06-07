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
        # provide default init pose to avoid crashes
        init_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) if init_pose is None else init_pose

        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # it's ok for us to deviate from x y target pos by this distance
        self.tolerable_xy_dev = eucl_distance([0., 0.], [10., 10.])

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

        if self.sim.init_pose[2] >= self.target_pos[2]:
            raise ValueError("Target z position must be higher than initial!")

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        crash_penalty = self.get_crash_penalty()

        if self.reached_target():
            rew_z = 5.
        else:
            rew_z = self.get_z_reward() if self.sim.pose[2] > self.sim.init_pose[2] else self.get_z_penalty()
            # multiply by coefficient to stay focused on elevation
            rew_z *= 3.

        xy_deviation = eucl_distance(self.sim.pose[:2], self.target_pos[:2])
        relative_deviation = xy_deviation / self.tolerable_xy_dev
        # scale to 1 and multiply by 0.5 because for this particular task it's OK that agent has x and y axes deviations
        xy_deviation_penalty = np.tanh(relative_deviation)

        ang_penalties = np.tanh(abs(self.sim.pose[3]))
        ang_penalties += np.tanh(abs(self.sim.pose[4]))
        ang_penalties += np.tanh(abs(self.sim.pose[5]))

        return rew_z - xy_deviation_penalty - ang_penalties - crash_penalty

    def get_z_penalty(self):
        # current distance is height below the start point
        # max distance is start point height or 0.001 if we started from the ground
        return -self.reward_function(abs(min(self.sim.pose[2] - self.sim.init_pose[2], 0)),
                                     max(self.sim.init_pose[2], 0.001))

    def get_z_reward(self):
        # current distance is height above the start point
        # max distance is how much we need to rise above the start point
        return 1. - self.reward_function(np.abs(self.sim.pose[2] - self.target_pos[2]),
                                         np.abs(self.target_pos[2] - self.sim.init_pose[2]))

    def reward_function(self, curr_distance, max_distance):
        return (curr_distance / max_distance) ** 0.4

    def reached_target(self):
        return self.sim.pose[2] >= self.target_pos[2]

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
            # if self.reached_target():
            #     done = True
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state

    def get_crash_penalty(self):
        for i in range(3):
            if self.sim.pose[i] <= self.sim.lower_bounds[i] or self.sim.pose[i] > self.sim.upper_bounds[i]:
                return 100.
        return 0.

    def get_torques(self):
        return [ang_accel * mom_of_inertia for ang_accel, mom_of_inertia in
                zip(self.sim.angular_accels, self.sim.moments_of_inertia)]


def eucl_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
