import torch

a=torch.arange(12)

print(a)

b=a.reshape(3,4)

print(b)

a1=a.reshape(2,6)
a2=a.reshape(6,2)
a3=a.reshape(4,3)
a4=a.reshape(3,2,2)

print(a1,'\n',a2,'\n',a3,'\n',a4)