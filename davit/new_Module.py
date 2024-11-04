import time
import sys
import torch
import torch.nn as nn
import numpy as np
from lib.layer_utils import list2vec, vec2list, norm_diff, conv3x3, conv5x5
from scipy.io import loadmat
from PIL import Image
from collections import deque
import torch.nn.functional as F
from collections import OrderedDict
from config import config
from tools.image_utils import rgb2ycbcr, calc_psnr, calc_ssim
from network.network import ADMM_RED_UNFOLD
from config import update_config
from lib.layer_utils import conv3x3, conv5x5
from lib.optimizations import VariationalHidDropout2d
from argparse import ArgumentParser
from torchvision import transforms
from torch.utils.data import DataLoader
from lib.solvers import anderson, broyden
from new_LSH import NonLocalSparseAttention
import argparse
from setting import Setting
import utils
from attention import Attention
from swinIR import SwinIR

DEQ_EXPAND = 5
NUM_GROUPS = 4  # 4->2 swinIR
BLOCK_GN_AFFINE = True


class DownsampleModule(nn.Module):
    def __init__(self, num_channels, in_res, out_res):
        """
        A downsample step from resolution j (with in_res) to resolution i (with out_res). A series of 2-strided convolutions.
        """
        super(DownsampleModule, self).__init__()
        # downsample (in_res=j, out_res=i)
        convs = []
        inp_chan = num_channels[in_res]
        out_chan = num_channels[out_res]
        self.level_diff = level_diff = out_res - in_res

        kwargs = {"kernel_size": 3, "stride": 2, "padding": 1, "bias": False}
        for k in range(level_diff):
            intermediate_out = out_chan if k == (level_diff - 1) else inp_chan
            components = [('conv', nn.Conv2d(inp_chan, intermediate_out, **kwargs)),
                          ('gnorm', nn.GroupNorm(NUM_GROUPS, intermediate_out, affine=FUSE_GN_AFFINE))]
            if k != (level_diff - 1):
                components.append(('relu', nn.ReLU(inplace=True)))
            convs.append(nn.Sequential(OrderedDict(components)))
        self.net = nn.Sequential(*convs)

    def forward(self, x):
        return self.net(x)


class UpsampleModule(nn.Module):
    def __init__(self, num_channels, in_res, out_res):
        """
        An upsample step from resolution j (with in_res) to resolution i (with out_res).
        Simply a 1x1 convolution followed by an interpolation.
        """
        super(UpsampleModule, self).__init__()
        # upsample (in_res=j, out_res=i)
        inp_chan = num_channels[in_res]
        out_chan = num_channels[out_res]
        self.level_diff = level_diff = in_res - out_res

        self.net = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(inp_chan, out_chan, kernel_size=1, bias=False)),
            ('gnorm', nn.GroupNorm(NUM_GROUPS, out_chan, affine=FUSE_GN_AFFINE)),
            ('upsample', nn.Upsample(scale_factor=2 ** level_diff, mode='nearest'))]))

    def forward(self, x):
        return self.net(x)


class DeBlocker(nn.Module):

    def __init__(self, n_channel):
        super(DeBlocker, self).__init__()

        # self.conv1 = nn.Conv2d(32, n_channel, 3, 1, 1, bias=True)
        # self.conv2 = nn.Conv2d(n_channel, n_channel, 3, 1, 1, bias=True)
        # self.conv3 = nn.Conv2d(n_channel, n_channel, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(n_channel, 1, 3, 1, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x, ):
        res = self.relu(self.conv4(x))
        return res


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        A bottleneck block with receptive field only 3x3. (This is not used in MDEQ; only
        in the classifier layer).
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM, affine=False)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM, affine=False)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, injection=None):
        if injection is None:
            injection = 0
        residual = x

        out = self.conv1(x) + injection
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class BranchNet(nn.Module):
    def __init__(self, blocks):
        """
        The residual block part of each resolution stream
        """
        super().__init__()
        self.blocks = blocks

    def forward(self, x, injection=None):
        blocks = self.blocks
        y = blocks[0](x, injection)
        for i in range(1, len(blocks)):
            y = blocks[i](y)
        return y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, n_big_kernels=0, dropout=0.0, wnorm=False):
        """
        A canonical residual block with two 3x3 convolutions and an intermediate ReLU. Corresponds to Figure 2
        in the paper.
        """
        super(BasicBlock, self).__init__()
        conv1 = conv3x3
        conv2 = conv3x3
        inner_planes = int(DEQ_EXPAND * planes)

        self.conv1 = conv1(inplanes, inner_planes)
        self.gn1 = nn.GroupNorm(NUM_GROUPS, inner_planes, affine=BLOCK_GN_AFFINE)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv2(inner_planes, planes)
        self.gn2 = nn.GroupNorm(NUM_GROUPS, planes, affine=BLOCK_GN_AFFINE)

        self.gn3 = nn.GroupNorm(NUM_GROUPS, planes, affine=BLOCK_GN_AFFINE)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.drop = VariationalHidDropout2d(dropout)

    def _reset(self, bsz, d, H, W):
        """
        Reset dropout mask and recompute weight via weight normalization
        """
        if 'conv1_fn' in self.__dict__:
            self.conv1_fn.reset(self.conv1)
        if 'conv2_fn' in self.__dict__:
            self.conv2_fn.reset(self.conv2)
        self.drop.reset_mask(bsz, d, H, W)

    def forward(self, x, injection=None):
        if injection is None: injection = 0
        residual = x

        out = self.relu(self.gn1(self.conv1(x)))
        out = self.drop(self.conv2(out)) + injection
        out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gn3(self.relu3(out))
        return out


blocks_dict = {'BASIC': BasicBlock}


class MDEQModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_channels, big_kernels, dropout=0.0):
        """
        An MDEQ layer (note that MDEQ only has one layer).
        """
        super(MDEQModule, self).__init__()
        self.num_branches = num_branches
        self.num_channels = num_channels
        self.big_kernels = big_kernels
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels, big_kernels,
                                            dropout=dropout)

        # self.attentions = nn.ModuleList()
        # for _ in num_channels:
        #     self.attentions.append(Attention(num_channels[-1] // 6))
        self.fuse_layers = self._make_fuse_layers()
        self.post_fuse_layers = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('relu', nn.ReLU(False)),
                ('conv', nn.Conv2d(num_channels[i], num_channels[i], kernel_size=1, bias=False)),
                ('gnorm', nn.GroupNorm(NUM_GROUPS // 2, num_channels[i], affine=POST_GN_AFFINE))
            ])) for i in range(num_branches)])

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, big_kernels, stride=1, dropout=0.0):
        """
        Make a specific branch indexed by `branch_index`. This branch contains `num_blocks` residual blocks of type `block`.
        """
        layers = nn.ModuleList()
        n_channel = num_channels[branch_index]
        n_big_kernels = big_kernels[branch_index]
        for i in range(num_blocks[branch_index]):
            layers.append(block(n_channel, n_channel, n_big_kernels=n_big_kernels, dropout=dropout))
        return BranchNet(layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels, big_kernels, dropout=0.0):
        """
        Make the residual block (s; default=1 block) of MDEQ's f_\theta layer. Specifically,
        it returns `branch_layers[i]` gives the module that operates on input from resolution i.
        """
        branch_layers = [self._make_one_branch(i, block, num_blocks, num_channels, big_kernels, dropout=dropout) for i
                         in range(num_branches)]
        return nn.ModuleList(branch_layers)

    def get_num_inchannels(self):
        return self.num_channels

    def _make_fuse_layers(self):
        """
        Create the multiscale fusion layer (which does simultaneous up- and downsamplings).
        """
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_channels = self.num_channels
        fuse_layers = []
        for i in range(num_branches):
            fuse_layer = []  # The fuse modules into branch #i
            for j in range(num_branches):
                if i == j:
                    fuse_layer.append(None)  # Identity if the same branch
                else:
                    module = UpsampleModule if j > i else DownsampleModule
                    fuse_layer.append(module(num_channels, in_res=j, out_res=i))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        # fuse_layers[i][j] gives the (series of) conv3x3s that convert input from branch j to branch i
        return nn.ModuleList(fuse_layers)

    def _reset(self, xs):
        """
        Reset the dropout mask and the learnable parameters (if weight normalization is applied)
        """
        for i, branch in enumerate(self.branches):
            for block in branch.blocks:
                block._reset(*xs[i].shape)
            if 'post_fuse_fns' in self.__dict__:
                self.post_fuse_fns[i].reset(
                    self.post_fuse_layers[i].conv)  # Re-compute (...).conv.weight using _g and _v

    def forward(self, x, injection, *args):
        """
        The two steps of a multiscale DEQ module (see paper): a per-resolution residual block and
        a parallel multiscale fusion step.
        """
        if injection is None:
            injection = [0] * len(x)
        if self.num_branches == 1:
            return [self.branches[0](x[0], injection[0])]

        # Step 1: Per-resolution residual block
        x_block = []
        for i in range(self.num_branches):
            x_block.append(self.branches[i](x[i], injection[i]))

        # Step 2: Multiscale fusion
        x_fuse = []
        for i in range(self.num_branches):
            y = 0
            # y_list = []
            # Start fusing all #j -> #i up/down-samplings
            for j in range(self.num_branches):
                y += x_block[j] if i == j else self.fuse_layers[i][j](x_block[j])
                # y_list.append(x_block[j] if i == j else self.fuse_layers[i][j](x_block[j]))
            # y = torch.stack(y_list, dim=1)
            # y = self.attentions[i](y)
            x_fuse.append(self.post_fuse_layers[i](y))
        return x_fuse


class Module_fff(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(Module_fff, self).__init__()
        global BN_MOMENTUM
        BN_MOMENTUM = 0.1
        self.parse_cfg(cfg)
        init_chansize = self.init_chansize
        self.change_channel = nn.Sequential(
            conv3x3(1, init_chansize, stride=(2 if self.downsample_times >= 1 else 1)),
            nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM, affine=True),
            nn.ReLU(inplace=True),
            conv3x3(init_chansize, init_chansize, stride=(2 if self.downsample_times >= 2 else 1)),
            nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM, affine=True),
            nn.ReLU(inplace=True))
        self.stage0 = None
        self.fullstage = self._make_stage(self.fullstage_cfg, self.num_channels, dropout=self.dropout)
        self.alternative_mode = "abs"
        self.iodrop = VariationalHidDropout2d(0.0)
        self.hook = None

        self.downsamples = nn.ModuleList()
        for j in range(1, self.num_branches):
            self.downsamples.append(DownsampleModule(self.num_channels, 0, j))

    def _make_stage(self, layer_config, num_channels, dropout=0.0):
        """
        Build an MDEQ block with the given hyperparameters
        """
        num_modules = 1
        num_branches = self.num_branches  # 2->4
        num_blocks = [1, 1, 1, 1]
        block_type = BasicBlock
        big_kernels = [0, 0, 0, 0]
        return MDEQModule(num_branches, block_type, num_blocks, num_channels, big_kernels, dropout=dropout)

    def parse_cfg(self, cfg):
        """
        Parse a configuration file
        """
        global DEQ_EXPAND, NUM_GROUPS, BLOCK_GN_AFFINE, FUSE_GN_AFFINE, POST_GN_AFFINE
        self.num_branches = 4  # 2->4
        self.init_chansize = 12  # 12 -> 6
        self.num_channels = [self.init_chansize, self.init_chansize, self.init_chansize,
                             self.init_chansize]  # [12,12]->[12,12,12,12]
        self.num_layers = 4
        self.dropout = 0.25
        self.wnorm = True
        self.num_classes = 10
        self.downsample_times = 0
        self.fullstage_cfg = cfg['MODEL']['EXTRA']['FULL_STAGE']
        self.pretrain_steps = 3000

        # DEQ related
        self.f_solver = eval(cfg['DEQ']['F_SOLVER'])
        self.b_solver = eval(cfg['DEQ']['B_SOLVER'])
        self.lsh = NonLocalSparseAttention()
        # self.swinir = SwinIR(upscale=1, img_size=(96, 96), in_chans=self.init_chansize,
        #                      window_size=8, img_range=1., depths=[3, 3, 3, 3],
        #                      embed_dim=30, num_heads=[3, 3, 3, 3], mlp_ratio=2, upsampler='pixelshuffledirect')
        self.f_thres = 18
        self.b_thres = 20
        self.stop_mode = 'rel'

        # Update global variables
        DEQ_EXPAND = 5
        NUM_GROUPS = 4  # 4->2 swinIR
        BLOCK_GN_AFFINE = True
        FUSE_GN_AFFINE = True
        POST_GN_AFFINE = False

    def _forward(self, x, train_step=-1, compute_jac_loss=True, **kwargs):
        """
        The core MDEQ module. In the starting phase, we can (optionally) enter a shallow stacked f_\theta training mode
        to warm up the weights (specified by the self.pretrain_steps; see below)
        """
        num_branches = self.num_branches
        f_thres = kwargs.get('f_thres', self.f_thres)
        b_thres = kwargs.get('b_thres', self.b_thres)
        x = self.change_channel(x)

        # Inject only to the highest resolution...
        x = self.lsh(x)
        # x = self.swinir(x)
        x_list = [self.stage0(x) if self.stage0 else x]
        for i in range(1, num_branches):
            bsz, _, H, W = x_list[-1].shape
            # x_list.append(
            #     torch.zeros(bsz, self.num_channels[i], H // 2, W // 2).to(x))  # ... and the rest are all zeros
            x_list.append(self.downsamples[i - 1](x_list[0]))

        z_list = [torch.zeros_like(elem) for elem in x_list]
        z1 = list2vec(z_list)
        cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z_list]

        func = lambda z: list2vec(self.fullstage(vec2list(z, cutoffs), x_list))

        # For variational dropout mask resetting and weight normalization re-computations
        self.fullstage._reset(z_list)

        jac_loss = torch.tensor(0.0).to(x)
        sradius = torch.zeros(bsz, 1).to(x)
        deq_mode = (train_step < 0) or (train_step >= self.pretrain_steps)

        # Multiscale Deep Equilibrium!
        if not deq_mode:
            for layer_ind in range(self.num_layers):
                z1 = func(z1)
            new_z1 = z1

            if self.training:
                if compute_jac_loss:
                    z2 = z1.clone().detach().requires_grad_()
                    new_z2 = func(z2)
        y_list = self.iodrop(vec2list(new_z1, cutoffs))
        return y_list, jac_loss.view(1, -1), sradius.view(-1, 1)


class MDEQClsNet(Module_fff):
    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ Classification model with the given hyperparameters
        """
        global BN_MOMENTUM

        super(MDEQClsNet, self).__init__(cfg, BN_MOMENTUM=0.1, **kwargs)
        self.head_channels = [8, 16, 16, 16]
        self.final_chansize = 200

        # Classification Head
        self.incre_modules, self.downsamp_modules = self._make_head(self.num_channels)
        self.classifier = nn.Linear(self.final_chansize, self.num_classes)

        self.channel = 32
        self.deblocker = DeBlocker(self.channel)

    def _make_head(self, pre_stage_channels):
        """
        Create a classification head that:
           - Increase the number of features in each resolution
           - Downsample higher-resolution equilibria to the lowest-resolution and concatenate
           - Pass through a final FC layer for classification
        """
        head_block = Bottleneck
        d_model = 24
        head_channels = [8, 16, 16, 16]

        # Increasing the number of channels on each resolution when doing classification.
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            # incre_module = self._make_layer(head_block, channels, head_channels[i], blocks=1, stride=1)
            incre_module = self._make_layer(head_block, channels, 8, blocks=1, stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # Downsample the high-resolution streams to perform classification
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion
            downsamp_module = nn.Sequential(conv3x3(self.init_chansize, out_channels, stride=2, bias=True),
                                            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                                            nn.ReLU(inplace=True))
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        return incre_modules, downsamp_modules

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = nn.Sequential(nn.Conv2d(self.init_chansize, 32, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []

        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def predict(self, y_list):
        """
        Given outputs at multiple resolutions, predict the class of the image
        """
        # Classification Head 128 32 32 32
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            img = self.incre_modules[i + 1](y_list[i + 1])
            img = F.interpolate(img, size=(y_list[0].shape[-2], y_list[0].shape[-1]), mode='bilinear')
            y = img + y

        y = self.deblocker(y)
        # y = self.final_layer(y)
        #
        # # Pool to a 1x1 vector (if needed)
        # if torch._C._get_tracing_state():
        #     y = y.flatten(start_dim=2).mean(dim=2)
        # else:
        #     y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
        # y = self.classifier(y)
        return y

    def forward(self, x, train_step=0, **kwargs):
        y_list, jac_loss, sradius = self._forward(x, train_step, **kwargs)
        return self.predict(y_list), jac_loss, sradius


def parse_args(parser):
    # parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str,
                        default='experiments/cifar/cls_mdeq_TINY.yaml')
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')
    parser.add_argument('--percent',
                        help='percentage of training data to use',
                        type=float,
                        default=1.0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
    parser.add_argument('--end_epoch', type=int, default=100, help='epoch number of end training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--sample_ratio', type=float, default=0.1, help='sample ratio')
    parser.add_argument('--finetune', type=str, default='', help='pretrained parameter path')
    args = parser.parse_args()
    update_config(config, args)
    return args


def image_padding(img):
    block_size = 32
    hei, wid = img.shape
    hei_blk = hei // 32
    wid_blk = wid // 32

    pad_img = img[:hei_blk * 32, :wid_blk * 32]

    return pad_img, hei_blk * 32, wid_blk * 32


def process_img(img, only_y=True):
    n_dim = img.ndim
    if n_dim < 3:
        return img
    else:
        if (img[:, :, 0] == img[:, :, 1]).all() and (img[:, :, 0] == img[:, :, 2]).all() and (
                img[:, :, 1] == img[:, :, 2]).all():
            return img[:, :, 0]
        else:
            img_y = rgb2ycbcr(img, only_y=True)
            return img_y


if __name__ == '__main__':
    def get_cls_net(config, **kwargs):
        global BN_MOMENTUM
        BN_MOMENTUM = 0.1
        model = MDEQClsNet(config, **kwargs)
        return model


    import os


    device = 'cuda'
    model = ADMM_RED_UNFOLD(0.1)
    model.to(device)
    input = torch.randn(2, 3, 32, 32).to(device)
    model(input)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    parser = ArgumentParser(description='CSRN')
    args = parse_args(parser)
    fff = eval('get_cls_net')(config)
    print("-----------------------")
    print(sum(torch.numel(parameter) for parameter in fff.parameters()))
    print("-----------------------")

    model = ADMM_RED_UNFOLD(args.sample_ratio)
    print(sum(torch.numel(parameter) for parameter in model.parameters()))
    model.to(device)
    model = nn.DataParallel(model)

    if args.finetune:
        print(f"pretrained parameter file loaded : {args.finetune}")
        pretrained_dict = torch.load(args.finetune, map_location='cpu')
        model.load_state_dict(pretrained_dict['model'])
        for name, param in model.named_parameters():
            if 'davit' in name:
                param.requires_grad = False

    config1 = Setting(args.sample_ratio)

    train_dataset = loadmat(config1.train_dataset_name)['train']
    transformer = transforms.Compose([transforms.ToTensor()])
    trainloader = DataLoader(dataset=train_dataset, batch_size=config1.batch, num_workers=50, shuffle=True)
    print("Finished constructing model!")

    optimizer = torch.optim.Adam(model.parameters(), config1.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config1.step, gamma=0.1)
    criterion = F.mse_loss

    best = {'idx_epoch': 0, 'psnr': 0}
    psnr_list = []
    ssim_list = []

    start_epoch = utils.auto_load_model(config1, model, optimizer, scheduler)

    print("start---")
    time_start = time.time()
    for idx_epoch in range(start_epoch, config1.epoch):
        # training stage
        model.train()

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Epoch: [{}]'.format(idx_epoch)
        print_freq = 100

        n_pic = 0
        train_loss = []
        for data_iter_step, data in enumerate(metric_logger.log_every(trainloader, print_freq, header)):

            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)

            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())

            metric_logger.update(loss=loss.mean())

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        utils.save_checkpoint(config1, idx_epoch, model, optimizer, scheduler)

        # val
        model.eval()
        psnrs = []
        ssims = []
        times = []
        filelist = os.listdir(config1.val_dataset_name)
        with torch.no_grad():
            for i in range(len(filelist)):
                if os.path.splitext(filelist[i])[1] in ['.tif', '.bmp', '.png', '.jpg']:
                    name = os.path.splitext(filelist[i])[0]
                    filepath = config1.val_dataset_name + filelist[i]
                    img_ori = Image.open(filepath)
                    img_ori = np.array(img_ori)
                    img_y = process_img(img_ori, only_y=True)
                    img_y, hei, wid = image_padding(img_y)
                    img = img_y / 255.0
                    img = torch.from_numpy(img)
                    img = img.type(torch.FloatTensor)
                    img = img.unsqueeze(0).unsqueeze(0)
                    img = img.to(device)
                    start_time = time.time()
                    reconstruction = model(img)
                    end_time = time.time()

                    reconstruction = reconstruction[0][0].cpu().data.numpy()
                    reconstruction = np.clip(reconstruction, 0, 1)
                    reconstruction *= 255

                    psnr = calc_psnr(np.array(np.round(reconstruction), dtype='uint8'), img_y)
                    ssim = calc_ssim(np.array(np.round(reconstruction), dtype='uint8'), img_y)
                    cal_time = end_time - start_time

                    # print(psnr, '  ', ssim, '---{}'.format(cal_time))

                    psnrs.append(psnr)
                    ssims.append(ssim)
                    times.append(cal_time)

        scheduler.step()
        print('mean_psnr = {}, mean_ssim = {}, mean_time = {}'.format(np.mean(psnrs), np.mean(ssims), np.mean(times)))

        mean_train_loss = np.mean(train_loss)
        mean_psnr = np.mean(psnrs)
        psnr_list.append(mean_psnr)
        mean_ssim = np.mean(ssims)
        ssim_list.append(mean_ssim)

        output_file = open(config1.log_file, 'a')
        output_file.write('[{}/{}] \nTrain Loss: {} \nBSDS500: psnr: {}ssim: {}'.format(idx_epoch + 1, config1.epoch,
                                                                                        np.mean(mean_train_loss),
                                                                                        np.mean(psnrs),
                                                                                        np.mean(ssims)) + '\n')

        if np.mean(mean_psnr) > best['psnr']:
            best['psnr'] = np.mean(mean_psnr)
            best['idx_epoch'] = idx_epoch
            utils.save_checkpoint(config1, 'best', model, optimizer, scheduler)

    output_file = open(config1.log_file, 'a')
    output_file.write('we aquire best val psnr at epoch {} -- {}.\n'.format(best['idx_epoch'], best['psnr']))
    time_final = time.time()
    print(time_final - time_start)
