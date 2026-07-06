import torch

#3x4크기의 랜덤 행렬 생성
a=torch.rand(3,4)

print(a)

#5x5크기의 랜덤 행렬 생성
rand_torch=torch.rand(5,5)

print(rand_torch)

#5x5크기의 0으로 이루어진 행렬 생성
zero_torch=torch.zeros(5,5)

print(zero_torch)

#5x5크기의 1으로 이루어진 행렬 생성
one_torch=torch.ones(5,5)

print(one_torch)

#5x5크기의 정규분포 행렬 생성

bell_torch=torch.randn(5,5)

print(bell_torch)