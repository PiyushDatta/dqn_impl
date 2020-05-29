import random
import numpy as np
from collections import namedtuple
from typing import List


class ReplayMemory(object):
  def __init__(self, capacity: int) -> None:
    self.capacity = capacity
    self.memory = []
    self.position = 0
    self.transition = namedtuple("Transition",
                                 field_names=["prev_state", "action",
                                              "reward", "curr_state",
                                              "done"])

  def push(self, prev_state: np.ndarray, action: int,
           reward: int, curr_state: np.ndarray, done: bool) -> None:
    if self.position < self.capacity:
      self.memory.append(self.transition(
          prev_state, action, reward, curr_state, done))
    else:
      self.memory[self.position] = self.transition(
          prev_state, action, reward, curr_state, done)

    self.position = (self.position+1) % self.capacity

  def sample(self, batch_size: int) -> List:
    return random.sample(self.memory, batch_size)

  def __len__(self) -> int:
    return len(self.memory)
