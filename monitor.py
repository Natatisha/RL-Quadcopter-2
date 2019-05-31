from collections import deque
import math
import numpy as np
import matplotlib.pyplot as plt
import csv


def interact(task, agent, num_episodes=10000, average_range=100, max_episode_lenght=500, window=100):
    # initialize average rewards
    avg_rewards = deque(maxlen=num_episodes)
    # initialize best average reward
    best_avg_reward = -math.inf
    # initialize monitor for most recent rewards
    samp_rewards = deque(maxlen=window)
    # initialize monitor for total rewards
    total_reward = 0

    # for each episode
    for i_episode in range(1, num_episodes + 1):
        # begin the episode
        state = agent.reset_episode()
        # initialize the sampled reward
        samp_reward = 0
        while True:
            # agent selects an action
            action = agent.act(state)
            # agent performs the selected action
            next_state, reward, done = task.step(action)
            # agent performs internal updates based on sampled experience
            agent.step(action, reward, next_state, done)
            # update the sampled reward
            samp_reward += reward
            # update the total reward
            total_reward += reward
            # update the state (s <- s') to next time step
            state = next_state
            if done:
                # save final sampled reward
                samp_rewards.append(samp_reward)
                break
        if i_episode % average_range == 0:
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards)
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
            # monitor progress
            print("\rEpisode {}/{} || Average reward {} || Best average reward {}"
                  .format(i_episode, num_episodes, avg_reward, best_avg_reward))
        if i_episode == num_episodes: print('\n')
    return total_reward, avg_rewards, best_avg_reward


def plot_rewards(avg_rewards):
    plt.figure(figsize=(20, 10))
    plt.xlabel('episodes')
    plt.ylabel('average reward')
    plt.plot(avg_rewards)
    plt.show()


def run_sample_task(agent, task, file_out='sample_data.txt'):
    labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
              'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
              'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
    results = {x: [] for x in labels}

    # Run the simulation, and save the results.
    with open(file_out, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)
        while True:
            state = agent.reset_episode()
            rotor_speeds = agent.act(state)
            _, _, done = task.step(rotor_speeds)
            to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(
                rotor_speeds)
            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])
            writer.writerow(to_write)
            if done:
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
