import torch
import torchvision
from model_utils import _number_of_params

#model = torchvision.models.resnet18()
model= torchvision.models.vgg16_bn()
# print(model)

# for name, p in model.named_parameters():
#     print(name)
#     print(p.shape)
#     print(_number_of_params(p))

m = torch.nn.Dropout(p=0.5)
input = torch.randn(20, 16)
output = m(input)
print(output)