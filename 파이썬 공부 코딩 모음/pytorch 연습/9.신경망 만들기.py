import torch.nn as nn

model=nn.Sequential(
    nn.Linear(4,8),
    nn.ReLU(),
    nn.Linear(8,2)
)

print(model)

test_model=nn.Sequential(
    nn.Linear(10,32),
    nn.ReLU(),
    nn.Linear(32,3)
)

print(test_model)