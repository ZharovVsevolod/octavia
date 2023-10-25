import torch

x = [torch.tensor([1, 2, 3, 4]), torch.tensor([4, 5, 6, 7]), torch.tensor([7, 8, 9, 7])]
print(x)

print()

x = torch.stack(x)

# x = torch.cat(x)
print(x)

