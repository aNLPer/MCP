import torch
import torch.nn as nn
import torch.nn.functional as F
# pool of square window of size=3, stride=2
m = nn.MaxPool2d(3, stride=2)
# pool of non-square window
m = nn.MaxPool2d((50, 1), stride=(2, 1))
input = torch.randn(20, 16, 50, 32)
output = F.max_pool2d(input,kernel_size=(50, 1), stride=(2, 1))
print("end")