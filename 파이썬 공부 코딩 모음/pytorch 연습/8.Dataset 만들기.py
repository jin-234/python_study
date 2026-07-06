from torch.utils.data import TensorDataset,DataLoader
import torch

x=torch.randn(100,5)

print(x)

y=torch.randint(0,2,(100,))
print(y)
dataset=TensorDataset(x,y)

loader=DataLoader(dataset,batch_size=16,shuffle=True)

for x,y in loader:
    print(x.shape)