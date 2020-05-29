import os
import gym
from gym.envs.classic_control import CartPoleEnv
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from collections import deque
from agent import Agent


def train_agent(env: gym.envs.classic_control.CartPoleEnv, train_agent: Agent, target_agent: Agent,
                progress_print_per_iter: int, total_episodes: int, episode_epsilon: float,
                min_epsilon: float, max_epsilon_episodes: int, epsilon_decay: float,
                copy_max_count: int, saved_results_path: str, saved_results_name: str,
                hyperparams_dict: Dict) -> None:
  """
    Train the agent on a number of games/episodes.
    Decay epsilon by epsilon_decay if given, otherwise decay using max_epsilon_episodes.
    Decay learning rate by learning_rate_decay if present, otherwise don't decay.
    Record and plot data on matplotlib and also save the figues/numbers.
  """
  total_rewards = 0
  total_loss = 0
  total_steps = 0
  total_bellman_eq = 0
  avg_reward = deque(maxlen=100)
  progress_bar = tqdm(total=total_episodes)
  plotting_data = {
      'avg_rewards[last_%s]' % avg_reward.maxlen: np.empty(total_episodes),
      'total_rewards': np.empty(total_episodes),
      'epsilon': np.empty(total_episodes),
      'loss': np.empty(total_episodes),
      'bellman_eq': np.empty(total_episodes),
      'learning_rate': np.empty(total_episodes),
  }

  for episode in range(total_episodes):
    # epsilon decay
    episode_epsilon = epsilon_decay_formula(episode=episode, max_episode=max_epsilon_episodes,
                                            min_epsilon=min_epsilon, epsilon=episode_epsilon,
                                            epsilon_decay=epsilon_decay)

    # train game/episode, save weights, decay learning rate
    total_rewards, total_loss, total_steps, total_bellman_eq = train_single_game(env=env,
                                                                                 train_agent=train_agent,
                                                                                 target_agent=target_agent,
                                                                                 epsilon=episode_epsilon,
                                                                                 copy_max_count=copy_max_count,
                                                                                 total_steps=total_steps)
    train_agent.decay_learning_rate()
    train_agent.save_weights()

    # update matplotlib data
    avg_reward.append(total_rewards)
    plotting_data['avg_rewards[last_%s]' %
                  avg_reward.maxlen][episode] = np.mean(avg_reward)
    plotting_data['total_rewards'][episode] = total_rewards
    plotting_data['epsilon'][episode] = episode_epsilon
    plotting_data['loss'][episode] = total_loss
    plotting_data['bellman_eq'][episode] = total_bellman_eq
    plotting_data['learning_rate'][episode] = train_agent.get_last_lr()

    # update progress bar
    if ((episode+1) % progress_print_per_iter) == 0:
      progress_bar.update(progress_print_per_iter)
      progress_bar.set_postfix({
          'episode reward': total_rewards,
          'avg reward (last %s)' % avg_reward.maxlen: np.mean(avg_reward),
          'epsilon': episode_epsilon,
      })

  env.close()
  save_results(plotting_data=plotting_data, progress_bar=progress_bar,
               name=saved_results_name, directory_name=saved_results_path,
               hyperparams_dict=hyperparams_dict)


def train_single_game(env: gym.envs.classic_control.CartPoleEnv,
                      train_agent: Agent, target_agent: Agent,
                      epsilon: float, copy_max_count: int,
                      total_steps: int) -> Tuple[int, int, int, float]:
  """
    Train the agent on one game/episode.
    Update target agent model weights to the same as train agent's after some number of steps.
    Return the total rewards given by the environment, and the loss given by the agent/model.

    observation := [cart position, cart velocity, pole angle, pole velocity at tip]
  """
  prev_observation = env.reset()
  observation = None
  total_rewards = 0
  total_loss = 0
  avg_bellman_eq = 0
  total_bellman_eq = 0
  reward, game_done = None, False

  while not game_done:
    # Get our agent's action and record the environment
    action = train_agent.get_action(
        observation=prev_observation, epsilon=epsilon)
    observation, reward, game_done, _ = env.step(action)
    total_rewards += reward
    total_steps += 1

    if game_done:
      reward -= 1

    # Add the observations we got from the environment
    train_agent.add_experience(prev_observation, action, reward,
                               observation, game_done)
    # Get the loss and bellman equation values from training
    total_loss, avg_bellman_eq = train_agent.train(target_agent)
    total_bellman_eq = avg_bellman_eq if avg_bellman_eq > 0 else total_bellman_eq
    # adjust prev state to curr state for next iteration
    prev_observation = observation

    # copy weights of policy net to our target net after a certain amount of steps
    if total_steps % copy_max_count == 0:
      target_agent.copy_weights(train_agent)

  return total_rewards, total_loss, total_steps, total_bellman_eq


def save_results(plotting_data: Dict[str, np.ndarray], progress_bar: tqdm,
                 name: str, directory_name: str, hyperparams_dict: Dict) -> None:
  """
    Save the progress bar and the matplotlib figures to a directory.
  """
  # create all the necessary directories
  parent_dir = os.path.abspath(os.path.join(directory_name, os.pardir))
  if not os.path.exists(parent_dir):
    os.mkdir(parent_dir)
  if not os.path.exists(directory_name):
    os.mkdir(directory_name)
  directory_name = directory_name + "/" + name

  # plot all our data
  def plot_figure(data: np.ndarray, xlabel: str, ylabel: str, save_path: str) -> None:
    plt.clf()
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.show()

  for name, plot_data in plotting_data.items():
    plot_figure(data=plot_data, xlabel='Episode', ylabel=name,
                save_path=directory_name + name + '.png')

  # Save the hyperparams and progress bar in a text file
  with open(directory_name+'pbar.txt', 'w', encoding="utf-8") as filetowrite:
    filetowrite.write("==== Hyperparams: ====\n")
    for key, val in hyperparams_dict.items():
      filetowrite.write(key + ": %s" % val)
      filetowrite.write("\n")
    filetowrite.write("\n")
    filetowrite.write(str(progress_bar))


def play_game(env: gym.envs.classic_control.CartPoleEnv, agent: Agent,
              epsilon: float, game_render: bool = False) -> None:
  """
    Play a single game.

    observation := [cart position, cart velocity, pole angle, pole velocity at tip]
  """
  observation = env.reset()
  done = False
  total_episodes = 0

  while not done:
    if game_render:
      env.render()
    action = agent.get_action(observation=observation, epsilon=epsilon)
    observation, _, done, _ = env.step(action)
    total_episodes += 1

  env.close()
  print("\nTotal rewards/time steps: {0}.".format(total_episodes))


def epsilon_decay_formula(episode: int, max_episode: int, min_epsilon: float,
                          epsilon: float, epsilon_decay: float) -> float:
  """
    If there is an epsilon decay value, then we use that.

    Otherwise use max_epsilon_episodes, which will look like the graph below:
    Returns 𝜺-greedy
    1.0---|\
          | \
          |  \
    min_e +---+------->
              |
              max_episode
  """
  if epsilon_decay > 0:
    new_epsilon = max(min_epsilon, epsilon * epsilon_decay)
  else:
    slope = (min_epsilon - 1.0) / max_episode
    new_epsilon = max(min_epsilon, slope * episode + epsilon)

  return new_epsilon
