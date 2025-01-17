
import torch
from torch import nn


model = nn.Conv3d(3, 1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False)
inputs = torch.randn(2436, 3, 2, 14, 14)

outputs = model(inputs)
print(outputs)
print(outputs.view(-1, 1280))
print(outputs.size())
