from collections import deque
import math
import numpy as np
import matplotlib.pyplot as plt
import time


def interact(task, agent, num_episodes=10000, window=20):
    best_avg_reward = -math.inf
    best_reward = -math.inf

    episodes_rewards = deque(maxlen=num_episodes)
    samp_rewards = deque(maxlen=window)

    for i_episode in range(1, num_episodes + 1):
        state = agent.reset_episode()
        episode_reward = 0
        while True:
            # agent selects an action
            action = agent.act(state)
            next_state, reward, done = task.step(action)
            # agent performs the selected action
            # agent performs internal updates based on sampled experience
            agent.step(action, reward, next_state, done)
            # update the sampled reward
            episode_reward += reward
            # update the state (s <- s') to next time step
            state = next_state
            if done:
                episodes_rewards.append(episode_reward)
                samp_rewards.append(episode_reward)
                break

        # update best average reward
        if episode_reward > best_reward:
            best_reward = episode_reward

        if i_episode >= 100:
            # get average reward from last 150 episodes
            avg_reward = np.mean(samp_rewards)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward

        if best_avg_reward >= 900.:
            print("\nTask accomplished in {} episodes.".format(i_episode), end="")
            break

        # monitor progress
        print("\rEpisode {}/{} || Reward {} || Best reward {} \nQuadcopter pose {}"
              .format(i_episode, num_episodes, episode_reward, best_reward, task.sim.pose[:3], end=""))
        if i_episode == num_episodes: print('\n')
    return episodes_rewards, best_reward


def plot_rewards(avg_rewards):
    plt.figure(figsize=(20, 10))
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.plot(avg_rewards)
    plt.show()


def run_sample_task(agent, task):
    labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
              'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
              'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
    results = {x: [] for x in labels}

    # animated_plot = AnimatedPlot()
    state = agent.reset_episode()
    total_reward = 0
    while True:
        #   animated_plot.plot(task)
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(action)
        for ii in range(len(labels)):
            results[labels[ii]].append(to_write[ii])
        total_reward += reward
        state = next_state
        if done:
            print("Total episode reward : {}".format(total_reward))
            break
    return results


def plot_position(results):
    plt.plot(results['time'], results['x'], label='x')
    plt.plot(results['time'], results['y'], label='y')
    plt.plot(results['time'], results['z'], label='z')
    plt.legend()
    _ = plt.ylim()


def plot_velocity(results):
    plt.plot(results['time'], results['x_velocity'], label='x_hat')
    plt.plot(results['time'], results['y_velocity'], label='y_hat')
    plt.plot(results['time'], results['z_velocity'], label='z_hat')
    plt.legend()
    _ = plt.ylim()


def plot_euler_angles(results):
    plt.plot(results['time'], results['phi'], label='phi')
    plt.plot(results['time'], results['theta'], label='theta')
    plt.plot(results['time'], results['psi'], label='psi')
    plt.legend()
    _ = plt.ylim()


def plot_euler_angles_velocities(results):
    plt.plot(results['time'], results['phi_velocity'], label='phi_velocity')
    plt.plot(results['time'], results['theta_velocity'], label='theta_velocity')
    plt.plot(results['time'], results['psi_velocity'], label='psi_velocity')
    plt.legend()
    _ = plt.ylim()


def plot_choice_of_actions(results):
    plt.plot(results['time'], results['rotor_speed1'], label='Rotor 1 revolutions / second')
    plt.plot(results['time'], results['rotor_speed2'], label='Rotor 2 revolutions / second')
    plt.plot(results['time'], results['rotor_speed3'], label='Rotor 3 revolutions / second')
    plt.plot(results['time'], results['rotor_speed4'], label='Rotor 4 revolutions / second')
    plt.legend()
    _ = plt.ylim()


class AnimatedPlot():
    def __init__(self):
        """Initialize parameters"""
        self.X, self.Y, self.Z = [], [], []

        self.fig = plt.figure(figsize=(14, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

    def plot(self, task, i_episode=None):
        pose = task.sim.pose[:3]
        self.X.append(pose[0])
        self.Y.append(pose[1])
        self.Z.append(pose[2])
        self.ax.clear()
        if i_episode:
            plt.title("Episode {}".format(i_episode))

        if len(self.X) > 1:
            self.ax.scatter(self.X[:-1], self.Y[:-1], self.Z[:-1], c='k', alpha=0.3)
        if task.sim.done and task.sim.runtime > task.sim.time:
            # Colision
            self.ax.scatter(pose[0], pose[1], pose[2], c='r', marker='*', linewidths=5)
        else:
            self.ax.scatter(pose[0], pose[1], pose[2], c='k', marker='s', linewidths=5)

        self.fig.canvas.draw()
        time.sleep(0.5)
