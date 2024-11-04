import torch
from torch.nn import functional as F

img = torch.randn(4,3,24,24)
print()
img = F.interpolate(img,scale_factor=(1,2),mode='bilinear')
print()