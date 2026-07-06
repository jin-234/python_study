import torch

a=torch.arange(16).reshape(4,4)

print(a)

print(a[0],'\n',a[1],'\n',a[-1],'\n',a[1:3,1:3])