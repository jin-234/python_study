import torch

a=torch.tensor([1,2,3])
b=torch.tensor([4,5,6])

print(f'+:{a+b}\n-:{a-b}\nx:{a*b}\n/:{a/b}')

test_a=torch.tensor([
    [1,2],
    [3,4]
])

test_b=torch.tensor([
    [5,6],
    [7,8]
])

print(f'+:{test_a+test_b}\n-:{test_a-test_b}\nx:{test_a*test_b}\n행렬곱:{torch.matmul(test_a,test_b)}')
