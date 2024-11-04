# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import models
from cls_function import train
# from utils import get_optimizer
from models.mdeq import MDEQClsNet
from config import config
from models.mdeq import get_cls_net
from config import update_config

# Press the green button in the gutter to run the script.
def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
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

    args = parser.parse_args()
    update_config(config, args)

    return args
if __name__ == '__main__':
    input = torch.randn(64,3,32,32)
    print(config.MODEL.NAME)
    args = parse_args()
    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(config)
    print(sum(torch.numel(parameter) for parameter in model.parameters()))

    train_loader = [];
    optimizer = get_optimizer(config, model)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_loader) * config.TRAIN.END_EPOCH, eta_min=1e-6)

    train(config, input, model,lr_scheduler)

    print("      ")

