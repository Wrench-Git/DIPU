import torch
import torch_dipu

a = torch.Tensor([1, 2])
a_xpu = a.cuda()
print(a_xpu.cpu())

b = torch.Tensor([[1, 2], [3, 4]])
b_xpu = b.cuda()
print(b_xpu.cpu())

c = torch.Tensor([1, 2])
c_xpu = c.cuda()
print(c_xpu.cpu())

p = torch.addmv(a, b, c)
p_xpu = torch.addmv(a_xpu, b_xpu, c_xpu)
print(p_xpu.cpu())
assert torch.allclose(p_xpu.cpu(), p)

