import os
import torch
import time
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
from PIL import Image
from network.network import ADMM_RED_UNFOLD
from tools.image_utils import rgb2ycbcr, calc_psnr, calc_ssim
from config import config
from argparse import ArgumentParser
from config import update_config
from new_Module import MDEQClsNet

def image_padding(img):
    block_size = 32
    hei, wid = img.shape
    hei_blk = hei // 32
    wid_blk = wid // 32

    pad_img = img[:hei_blk * 32, :wid_blk * 32]

    return pad_img, hei_blk * 32, wid_blk * 32


def image_depadding(img, hei_ori, wid_ori):
    img = img[:, :, :hei_ori, :wid_ori]

    return img


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


def get_cls_net(config, **kwargs):
    global BN_MOMENTUM
    BN_MOMENTUM = 0.1
    model = MDEQClsNet(config, **kwargs)
    # 打印model
    print(model)
    return model


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
    parser.add_argument('--sample_ratio', type=float, default=0.5, help='sample ratio')
    args = parser.parse_args()
    update_config(config, args)
    return args


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = ArgumentParser(description='redprotest')
args = parse_args(parser)
sample_ratio = 0.1
modelpath = './results/{}_8e-05/model/checkpoint-best.pth'.format(
    sample_ratio)
logpath = './log'
if not os.path.exists(logpath):
    os.makedirs(logpath)

datasetpath = '/home/zhouyu/xiewei/compressed sensing/testset/'
picpath = './pic/'.format(sample_ratio)
if not os.path.exists(picpath):
    os.makedirs(picpath)
device = 'cuda'

para = torch.load(modelpath)['model']
new_state_dict = {}
for key, value in para.items():
    if key.startswith('module.'):
        new_key = key[7:]  # 去掉 'module.' 前缀
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value
para = new_state_dict

fff = eval('get_cls_net')(config)

model = ADMM_RED_UNFOLD(sample_ratio)

model.load_state_dict(para)
model.to(device)

# datasets = ['Set5', 'Set11', 'Set14', 'BSDS100', 'Urban100', 'DIV2K_valid_HR']
datasets = ['Set5', 'Set11', 'Set14']

psnr_list = []
ssim_list = []
time_list = []

for dataset in datasets:
    dataset_path = datasetpath + dataset + '/'
    if os.path.exists(dataset_path):
        rootpath = picpath + str(sample_ratio) + '/'
        if not os.path.exists(rootpath):
            os.makedirs(rootpath)
        rootpath = rootpath + dataset + '/'
        if not os.path.exists(rootpath):
            os.makedirs(rootpath)

        filelist = os.listdir(dataset_path)
        with torch.no_grad():
            for i in range(len(filelist)):
                if os.path.splitext(filelist[i])[1] in ['.tif', '.bmp', '.png', '.jpg']:
                    name = os.path.splitext(filelist[i])[0]
                    filepath = dataset_path + filelist[i]
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
                    prediction = model(img)
                    end_time = time.time()
                    time_consume = (end_time - start_time)
                    print(time_consume)

                    if name == '1':
                        continue

                    prediction = prediction.cpu().data.numpy()
                    prediction = np.clip(prediction, 0, 1)
                    prediction *= 255

                    temp_psnr = calc_psnr(np.array(np.round(prediction[0][0]), dtype='uint8'), img_y)
                    temp_ssim = calc_ssim(np.array(np.round(prediction[0][0]), dtype='uint8'), img_y)

                    print(temp_psnr, '  ', temp_ssim)

                    pic_save_path = rootpath + name + '_{}_{}'.format(temp_psnr, temp_ssim) + '.png'
                    img = Image.fromarray(np.array(np.round(prediction[0][0]), dtype='uint8'))
                    img.save(pic_save_path)

                    psnr_list.append(temp_psnr)
                    ssim_list.append(temp_ssim)
                    time_list.append(time_consume)

            with open(os.path.join(logpath, f'{sample_ratio}.txt'), 'a+') as f:
                f.write(dataset + '\n')
                f.write('psnr:' + str(np.mean(psnr_list).round(2)) + '\n')
                f.write('ssim:' + str(np.mean(ssim_list).round(4)) + '\n')
                f.write('mean_times:' + str(np.mean(time_list).round(2)) + '\n\n')

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

with open(os.path.join(logpath, f'{sample_ratio}.txt'), 'a+') as f:
    f.write(str(total_params) + 'total parameters.' + '\n')
