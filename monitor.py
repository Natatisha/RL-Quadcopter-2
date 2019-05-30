from collections import deque
import sys
import math
import numpy as np
import matplotlib.pyplot as plt


def interact(task, agent, num_episodes=10000, max_episode_lenght=500, window=100):
    """ Monitor agent's performance.

    Params
    ======
    - env: instance of OpenAI Gym's Taxi-v1 environment
    - agent: instance of class Agent (see Agent.py for details)
    - num_episodes: number of episodes of agent-environment interaction
    - window: number of episodes to consider when calculating average rewards

    Returns
    =======
    - avg_rewards: deque containing average rewards
    - best_avg_reward: largest value in the avg_rewards deque
    """
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
        if (i_episode >= 100):
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards)
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
        # monitor progress
        print("\rEpisode {}/{} || Best average reward {}".format(i_episode, num_episodes, best_avg_reward), end="")
        sys.stdout.flush()
        # check if task is solved (according to OpenAI Gym)
        if best_avg_reward >= 9.7:
            print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
            break
        if i_episode == num_episodes: print('\n')
    return total_reward, avg_rewards, best_avg_reward


def plot_rewards(avg_rewards):
    plt.figure(figsize=(20, 10))
    plt.xlabel('episodes')
    plt.ylabel('average reward')
    plt.plot(avg_rewards)
    plt.show()
