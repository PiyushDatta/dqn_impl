import os
import gym
from gym.envs.classic_control import CartPoleEnv
import torch
import numpy as np
from typing import List, Dict, Tuple
from model import Model
from replay_memory import ReplayMemory


class Agent():
  def __init__(self, env: gym.envs.classic_control.CartPoleEnv, debug: bool, checkpoint_path: str,
               hidden_layer_size: int, replay_memory_cap: int, batch_size: int,
               learning_rate: float, learning_rate_decay: float, discount_factor: float) -> None:
    """
      Agent class that is responsible for training our neural network and overall 
      managing the DQN.
    """
    self.env = env
    self.num_states = env.observation_space.shape[0]
    self.num_actions = env.action_space.n

    self.debug = debug

    # if gpu is to be used
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.device(self.device)
    # if debug:
    print("Using device: %s" % self.device)

    self.__model = Model(num_states=self.num_states, num_actions=self.num_actions,
                         hidden_layer_size=hidden_layer_size).to(self.device)

    self.learning_rate_decay = learning_rate_decay
    self.optimizer = torch.optim.Adam(
        self.__model.parameters(), lr=learning_rate)
    self.__scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=self.optimizer, gamma=learning_rate_decay)
    self.loss_function = torch.nn.MSELoss()
    self.checkpoint_path = checkpoint_path
    self.discount_factor = discount_factor

    self.replay_memory = ReplayMemory(replay_memory_cap)
    self.batch_size = batch_size

  @property
  def model(self) -> Model:
    return self.__model

  def preprocess_observation(self, observation: np.ndarray) -> torch.Tensor:
    return torch.autograd.Variable(torch.Tensor(observation).to(self.device))

  def predict(self, input_data: np.ndarray) -> torch.Tensor:
    processed_data = self.preprocess_observation(
        input_data.reshape(-1, self.num_states))
    self.__model.train(mode=False)
    return self.__model(processed_data)

  def get_action(self, observation: np.ndarray, epsilon: float) -> int:
    """
      Only 2 actions, left and right. 
      0 = move black box thing left.
      1 = move black box thing right

      args:
        observation: List of floats. List[cart position, cart velocity, pole angle, pole velocity at tip].
    """
    if np.random.rand() < epsilon:
      final_action = self.env.action_space.sample()
    else:
      self.__model.train(mode=False)
      scores = self.predict(observation)
      _, max_val_idx = torch.max(scores.cpu().data, 1)
      final_action = int(max_val_idx.numpy())

    return final_action

  def decay_learning_rate(self) -> None:
    if len(self.replay_memory) >= self.batch_size and self.learning_rate_decay > 0.00:
      self.__scheduler.step()

  def get_last_lr(self) -> float:
    return self.__scheduler.get_last_lr()[0]

  def save_weights(self) -> None:
    # make the file if not exists, torch.save doesn't work without existing file
    try:
      if not os.path.exists(self.checkpoint_path):
        if not os.path.isdir(os.path.dirname(self.checkpoint_path)):
          os.mkdir(os.path.dirname(self.checkpoint_path))
        with open(self.checkpoint_path, 'w+'):
          pass

      if self.debug:
        print("Saving weights to: " + str(self.checkpoint_path))

      torch.save(self.__model.state_dict(), self.checkpoint_path)
    except Exception as e:
      print("Could not save weights to: " + str(self.checkpoint_path))
      print("ERROR: %s" % e)

  def load_weights(self, name: str = None) -> None:
    try:
      self.__model.load_state_dict(torch.load(self.checkpoint_path))
      if self.debug:
        print("Loaded weights for " + name +
              ", from: " + str(self.checkpoint_path))
    except Exception as e:
      print("Could not load weights for " + name +
            ", from: " + str(self.checkpoint_path))
      print("ERROR: %s" % e)

  def copy_weights(self, agent_to_copy: 'Agent') -> None:
    self.__model.load_state_dict(agent_to_copy.model.state_dict())

  def add_experience(self, prev_state: np.ndarray, action: int,
                     reward: int, curr_state: np.ndarray, done: bool) -> None:
    self.replay_memory.push(prev_state, action, reward, curr_state, done)

  def train(self, target_agent: 'Agent') -> Tuple[float, float]:
    """
      Train on a single game. Only train if our replay memory has enough saved memory, 
      which should be >= batch size.

      We take a minibatch (of size batch_size) from our replay memory. We use our 
      train_agent (policy network) to predict the Q values for the previous states.
      We use our target_agent (target network) to predict the Q values for the 
      current states, but we use these Q values from target_agent in our bellman
      equation to get the real Q values. Finally, we compare the Q values from
      the policy network with the Q values we get from the bellman equation.
    """
    # only start training process when we have enough experiences in the replay
    if len(self.replay_memory) < self.batch_size:
      return 0.00, 0.00

    # sample random batch from replay memory
    minibatch = self.replay_memory.sample(self.batch_size)
    prev_states = np.vstack([x.prev_state for x in minibatch])
    actions = np.array([x.action for x in minibatch])
    rewards = np.array([x.reward for x in minibatch])
    curr_states = np.vstack([x.curr_state for x in minibatch])
    dones = np.array([x.done for x in minibatch])

    # use train network to predict q values of prior states (before actual states)
    q_predict = self.predict(prev_states)

    # use bellman equation to get expected q-value of actual states
    q_target = q_predict.cpu().clone().data.numpy()
    q_curr_state_values = np.max(target_agent.predict(curr_states).cpu().data.numpy(),
                                 axis=1)
    bellman_eq = rewards + self.discount_factor * q_curr_state_values * ~dones
    q_target[np.arange(len(q_target)), actions] = bellman_eq
    q_target = self.preprocess_observation(q_target)

    # train our network based on the results from its
    # q_predict to expected values given by our target network (q_target)
    self.__model.train(mode=True)
    self.optimizer.zero_grad()
    loss = self.loss_function(q_predict, q_target)
    loss.backward()
    self.optimizer.step()
    return loss, np.mean(bellman_eq)
