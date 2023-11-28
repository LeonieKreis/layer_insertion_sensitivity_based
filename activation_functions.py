import torch
from torch import nn


# if you want to use a different epsilon in the activation function below,
# the change must also be made in the file model_utils.py in freeze parameters!
# it is hardcoded there!
class TanhLU_shifted(nn.Module):
    '''
    This activation function is in C²(R,R) and approximates the ReLU-function
    sigma(x)=x for x>=epsilon
    sigma(x)=epsilon*(tanh((x-epsilon)/epsilon) +1)for x<=epsilon

    '''

    def __init__(self, epsilon=0.01) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return torch.where(x >= self.epsilon, x,
                           self.epsilon*(torch.add(torch.tanh((x - self.epsilon)/self.epsilon), 1)))

    def __call__(self, x):
        return self.forward(x)
