from time import sleep
import torch
import torch.nn as nn

class Pre(nn.Module):

    def __init__(self, ratio, block_size = 32):
        super(Pre, self).__init__()
        self.block_size = block_size
        self.ratio = ratio

        # self.weight, self.weight_transpose = self.generate_matrix(self.block_size, self.ratio)
        self.sample = nn.Conv2d(1, round(self.ratio * self.block_size ** 2), self.block_size, self.block_size, bias=False)
        # self.sample.weight = nn.Parameter(self.weight, requires_grad=True)
        
        self.init = nn.Conv2d(round(self.ratio * self.block_size ** 2), self.block_size**2, 1, 1, bias=False)
        # self.init.weight = nn.Parameter(self.weight_transpose, requires_grad=True)
        self.pixelshuffle = nn.PixelShuffle(32)

    def forward(self, ori_x):

        output = []

        #采样
        y = self.sample(ori_x)

        #初始重构
        x = self.pixelshuffle(self.init(y))
        output.append(x)

        return output