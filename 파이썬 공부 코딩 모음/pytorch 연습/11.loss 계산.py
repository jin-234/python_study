import torch
from torch import nn
criterion=nn.CrossEntropyLoss()
pred=torch.randn(4,3)
label=torch.tensor([0,2,1,1])
loss=criterion(pred,label)
print(pred)
print(label)
print(loss)

criterion_test=nn.CrossEntropyLoss()
pred_test=torch.randn(8,3)
label_test=torch.tensor([0,1,1,2,1,0,2,1])

loss_test=criterion(pred_test,label_test)
print(loss_test)