import math

# lr = lr0 * e^(−kt)


def lr_decay_get_lr(lr0, decay, episodes):
  return lr0*math.exp(-1*decay*episodes)


def lr_decay_get_decay(lr0, lr, episodes):
  return math.log(lr/lr0)/(-1*episodes)


if __name__ == "__main__":
  lr0 = 0.001
  decay = 1.0
  episodes = 1
  lr = lr_decay_get_lr(lr0, decay, episodes)
  # decay = lr_decay_get_decay(lr0, lr, episodes)
  print(lr)
