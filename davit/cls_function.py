# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import numpy as np
import sys

import torch

sys.path.append("../")
# from utils import save_checkpoint, AverageMeter
import random
from tqdm import tqdm
from config import config
from config import update_config

logger = logging.getLogger(__name__)

        
def train(config, input, model,lr_scheduler):

    data_time = AverageMeter()

    model.train()

    end = time.time()


    data_time.update(time.time() - end)

    f_thres =18
    b_thres = 20
    factor = 0
    compute_jac_loss = (torch.rand([]).item() < config.LOSS.JAC_LOSS_FREQ) and (factor > 0)
    output, jac_loss, _ = model(input, train_step=(lr_scheduler._step_count - 1),
                                compute_jac_loss=compute_jac_loss,
                                f_thres=f_thres, b_thres=b_thres)
    # target = target.cuda(non_blocking=True)

    # compute gradient and do update step
    print(" ")


