import torch

device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

a=torch.rand(3,3)
b=torch.rand(3,3)

#둘다 같은 결과 나옴
a=a.to(device)

print(a)

b=b.cuda()

print(b)