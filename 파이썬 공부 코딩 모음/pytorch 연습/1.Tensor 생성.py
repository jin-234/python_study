import torch

#1. 스칼라
a=torch.tensor(3)

#2.벡터
b=torch.tensor([1,2,3])

#3행렬
c=torch.tensor([
    [1,2],
    [3,4]
])

print(a,b,c)

#4.3x3행렬
d=torch.tensor([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

print(d,d.shape)
