import torch
from torch import nn


# if you want to use a different epsilon in the activation function below,
# the change must also be made in the file model_utils.py in freeze parameters!
# it is hardcoded there!
class TanhLU_shifted(nn.Module):
    '''
    This activation function is in C²(R,R), which is needed for the lagrange multiplier theory
    sigma(x)=x for x>=0
    sigma(x)=tanh(x) for x<=0

    Hence the derivative
    sigma'(x)=1, x>=0
    sigma'(x)= sech²(x)=1/cosh(x)², x<=0
    '''

    def __init__(self, epsilon=0.01) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return torch.where(x >= self.epsilon, x,
                           self.epsilon*(torch.add(torch.tanh((x - self.epsilon)/self.epsilon), 1)))

    def __call__(self, x):
        return self.forward(x)
