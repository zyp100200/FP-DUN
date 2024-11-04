from math import inf
import torch
from torch.nn import init
import torch.nn as nn
from collections import deque
from tools import common
import torch.nn.functional as F
import numbers
from einops import rearrange
from swinIR import SwinIR
from davit.davit import DaViT
from lib.jacobian import jac_loss_estimate


class ADMM_RED_UNFOLD(nn.Module):

    def __init__(self, ratio, n_iteration=3, n_channel=32, block_size=32):
        super(ADMM_RED_UNFOLD, self).__init__()
        self.block_size = block_size
        self.ratio = ratio
        self.iteration = n_iteration
        self.channel = n_channel

        self.sample_weight = nn.Parameter(init.xavier_normal_(torch.Tensor(int(self.ratio * 1024), 3, 32, 32)))

        self.vblocks = nn.ModuleList([VBlock() for i in range(self.iteration)])

        self.fx = Fx(iteration=self.iteration)

        self.deblocker = DeBlocker(self.channel)

    @staticmethod
    def PhiTPhi_fun(x, y, PhiW):
        temp = F.conv2d(x, PhiW,padding=0, stride=32, bias=None) - y
        temp = F.conv_transpose2d(temp, PhiW, stride=32)
        return temp

    def forward(self, ori_x):
        PhiTb = F.conv2d(ori_x, self.sample_weight,stride=32, padding=0, bias=None)
        y = PhiTb
        x = F.conv_transpose2d(PhiTb, self.sample_weight, stride=32)
        # res = self.PhiTPhi_fun(x, y, self.sample_weight)
        #
        # for i in range(self.iteration):
        #     v = self.vblocks[i](x, res)
        #
        #     # 不定点找另一个点
        #     # z, _, _ = self.fff(v, train_step=0,
        #     #                    compute_jac_loss=False,
        #     #                    f_thres=18, b_thres=20)
        #     z = self.swinir(v)
        #     # for _ in range(4):
        #     #     z = self.swinir(z)
        #
        #     x = self.xblocks[i](v, z)
        #     res = self.PhiTPhi_fun(x, y, self.sample_weight)

        for i in range(self.iteration):
            v = self.vblocks[i](x, self.PhiTPhi_fun(x, y, self.sample_weight))
            x = self.fx(v, i)
            
        x = self.deblocker(x)

        return x


class Fx(nn.Module):
    def __init__(self, iteration=5):
        super(Fx, self).__init__()
        # self.swinir = SwinIR(upscale=1, img_size=(96, 96), in_chans=1,
        #                      window_size=8, img_range=1., depths=[3, 3, 3, 3],
        #                      embed_dim=30, num_heads=[3, 3, 3, 3], mlp_ratio=2, upsampler='pixelshuffledirect')
        self.davit = DaViT()
        self.xblocks = nn.ModuleList([XBlock() for i in range(iteration)])

    def forward(self, v, i):
        z = self.davit(v)
        x = self.xblocks[i](v, z)
        return x


class VBlock(nn.Module):

    def __init__(self):
        super(VBlock, self).__init__()

        self.para1 = nn.Parameter(torch.Tensor([0.1]))

    def forward(self, x, res):
        v = x - self.para1 * res

        return v


class XBlock(nn.Module):

    def __init__(self):
        super(XBlock, self).__init__()
        self.para2 = nn.Parameter(torch.Tensor([0.1]))

    def forward(self, v, z):
        xx = (1 - self.para2) * v + self.para2 * z
        return xx


class DeBlocker(nn.Module):

    def __init__(self, n_channel):
        super(DeBlocker, self).__init__()

        self.conv1 = nn.Conv2d(1, n_channel, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(n_channel, n_channel, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(n_channel, n_channel, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(n_channel, 1, 3, 1, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x, ):
        res = self.relu(self.conv1(x))
        res = self.relu(self.conv3(self.relu(self.conv2(res))))
        res = self.conv4(res)

        x = x + res

        return x


class Atten(torch.nn.Module):
    def __init__(self, channels):
        super(Atten, self).__init__()

        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)
        self.norm1 = LayerNorm(self.channels, 'WithBias')
        self.norm2 = LayerNorm(self.channels, 'WithBias')
        self.conv_q = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.conv_kv = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels * 2, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels * 2, self.channels * 2, kernel_size=3, stride=1, padding=1,
                      groups=self.channels * 2, bias=True)
        )
        self.conv_out = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1,
                                  bias=True)

    def forward(self, pre, cur):
        b, c, h, w = pre.shape
        pre_ln = self.norm1(pre)
        cur_ln = self.norm2(cur)
        q = self.conv_q(cur_ln)
        q = q.view(b, c, -1)
        k, v = self.conv_kv(pre_ln).chunk(2, dim=1)
        k = k.view(b, c, -1)
        v = v.view(b, c, -1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        att = torch.matmul(q, k.permute(0, 2, 1))
        att = self.softmax(att)
        out = torch.matmul(att, v).view(b, c, h, w)
        out = self.conv_out(out) + cur

        return out


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
