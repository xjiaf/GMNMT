import torch
# adjust these dims to match your attention shapes
B, H, T, d_k = 1, 4, 12, 32

q = torch.randn(B, H, T, d_k, device='cuda', dtype=torch.float32)
k = torch.randn(B, H, T, d_k, device='cuda', dtype=torch.float32)

# this is exactly what attention does
out = torch.matmul(q, k.transpose(-2, -1))
print("OK:", out.shape)
