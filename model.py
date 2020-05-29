import torch


class Model(torch.nn.Module):
  def __init__(self, num_states: int, num_actions: int, hidden_layer_size: int) -> None:
    """
      DQN using convolution neural network
      output matrix = floor((n + 2p - f)/s + 1)
      where n = input matrix size, p = padding, f = kernel size, s = stride
      ex. input = 7x7, p = 0, f = 3, s = 2
      = (7+2(0)-3)/2 + 1
      = (7+0-3)/2 + 1
      = 4/2 + 1
      = 3, so 3x3 is output matrix
    """
    super(Model, self).__init__()

    self.layer1 = torch.nn.Sequential(
        torch.nn.Linear(num_states, hidden_layer_size),
        torch.nn.BatchNorm1d(hidden_layer_size),
        torch.nn.PReLU()
    )

    self.layer2 = torch.nn.Sequential(
        torch.nn.Linear(hidden_layer_size, hidden_layer_size),
        torch.nn.BatchNorm1d(hidden_layer_size),
        torch.nn.PReLU()
    )

    self.final_layer = torch.nn.Linear(hidden_layer_size, num_actions)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.final_layer(x)
    return x
