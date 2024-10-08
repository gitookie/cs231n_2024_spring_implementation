import torch
import numpy as np

# N, S, T, H, Dim_head = 2, 5, 6, 3, 4
# query = torch.ones((N, S, H, Dim_head)).moveaxis(2, 1)   # (2, 3, 5, 4)
# key = torch.ones((N, T, H, Dim_head)).moveaxis(2, 1)     # (2, 3, 6, 4)
# out = torch.matmul(query, torch.transpose(key, 2, 3))
# print(out.shape)

# x = torch.ones((3, 4))
# x1 = x[:, 1:3].reshape(2, 3)
# x3 = x[1:3, :].view(4, 2)
# x4 = x[:, 1:3].contiguous().view(2, 3)
# x2 = x[:, 1:3].view(2, 3)


x = torch.sin(torch.ones((2, 3)))
print(x)
print(torch.arange(10).shape)

max_len = 100
embed_dim = 400
# torch.sin(torch.arange(max_len * 10000 ** (-torch.arange(0, embed_dim, 2) / embed_dim)))
pe = torch.sin(torch.arange(max_len)[:, None] * 10000 ** (-torch.arange(0, embed_dim, 2) / embed_dim)[None, :])
print(pe.shape)