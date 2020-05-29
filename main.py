import os
import gym
from gym.envs.classic_control import CartPoleEnv
from hyperparams import hyperparams_dict
from agent import Agent
from helper_funcs import play_game, train_agent


def print_all_hyperparams() -> None:
  """
    Print all the hyper parameters.
  """
  print()
  print("==== Hyperparams: ====")
  for key, val in hyperparams_dict.items():
    print(key + ": %s" % val)
  print()


def get_agent(env: gym.envs.classic_control.CartPoleEnv, agent_debug: bool) -> Agent:
  """
    Returns Agent class.
  """
  return Agent(env=env,
               debug=agent_debug,
               checkpoint_path=hyperparams_dict['checkpoint_path'],
               hidden_layer_size=hyperparams_dict['hidden_layer_size'],
               batch_size=hyperparams_dict['batch_size'],
               learning_rate=hyperparams_dict['learning_rate'],
               learning_rate_decay=hyperparams_dict['learning_rate_decay'],
               discount_factor=hyperparams_dict['discount_factor'],
               replay_memory_cap=hyperparams_dict['replay_memory_cap'])


def main(agent_debug: bool = False, train_model: bool = False) -> None:
  """
    Train on multiple episodes or play a single game.

    observation space (input size) = 4 --> env.observation_space.shape[0]
    action space (output size) = 2 --> env.action_space.n
  """
  my_env = gym.make('CartPole-v0')
  my_train_agent = get_agent(env=my_env, agent_debug=False)
  my_target_agent = get_agent(env=my_env, agent_debug=False)

  if os.path.exists(hyperparams_dict['checkpoint_path']):
    my_train_agent.load_weights(name="training agent")
    my_target_agent.load_weights(name="target agent")

  if train_model:
    print_all_hyperparams()
    train_agent(env=my_env, train_agent=my_train_agent, target_agent=my_target_agent,
                progress_print_per_iter=hyperparams_dict['progress_per_iteration'],
                total_episodes=hyperparams_dict['total_episodes'],
                episode_epsilon=hyperparams_dict['epsilon'],
                min_epsilon=hyperparams_dict['min_epsilon'],
                max_epsilon_episodes=hyperparams_dict['max_epsilon_episodes'],
                epsilon_decay=hyperparams_dict['epsilon_decay'],
                copy_max_count=hyperparams_dict['copy_max_step'],
                saved_results_path=hyperparams_dict['saved_results_path'],
                saved_results_name=hyperparams_dict['saved_results_name'],
                hyperparams_dict=hyperparams_dict)
  else:
    play_game(env=my_env, agent=my_train_agent, epsilon=0.00)


if __name__ == "__main__":
  gym.logger.set_level(40)

  main(train_model=True)
  # main()
