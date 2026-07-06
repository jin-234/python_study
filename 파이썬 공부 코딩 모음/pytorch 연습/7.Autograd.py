import torch

x=torch.tensor(2.0,requires_grad=True)

y=x**2+3*x+1

y.backward()
print(x.grad)

x=torch.tensor(3.0,requires_grad=True)

y=x**3+2*(x**2)+5
y.backward()
print(x.grad)