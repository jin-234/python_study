import torch
import torch.nn as nn

class Mynet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.input=nn.Linear(20,64)
        self.relu=nn.ReLU()
        self.hiddenlayer=nn.Linear(64,10)
    
    def forward(self,x):
        x=self.input(x)
        x=self.relu(x)
        x=self.hiddenlayer(x)
        return x
    
x=torch.randn(8,20)
model=Mynet()
print(model)
output=model(x)
print(output.shape)