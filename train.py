import math
import torch
import random
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm
import cv2
from tensorboardX import SummaryWriter
import os
from torch.utils.data.distributed import DistributedSampler
import argparse

# 可動部分
from dataloader import Train_Loader  # load data 用
from model.baseline_model import Baseline_Model  # load model 用
from loss import L1_Loss


cv2.setNumThreads(0)

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
args = parser.parse_args()
if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="NCCL", init_method='env://')  # 若跑不動可改成 gloo

# training seed
seed = 666 + args.local_rank  # 這邊改seed
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# hyperparameters
input_path = './dataset/NH-HAZE/train/haze'
gt_path = './dataset/NH-HAZE/train/gt'
model_name = 'baseline'  # 模型存檔名稱
check_point_path = '{}_{}'.format(model_name, 'weights')  # checkpoints 會存在這
if not os.path.isdir(check_point_path):
    os.mkdir(check_point_path)
crop_size = [256, 256]
start_epoch = 0
end_epoch = 1
check_point_epoch = 1
batch_size = 1
init_lr = 1e-4
min_lr = 1e-7
net = Baseline_Model()

writer = SummaryWriter(model_name)

if os.path.exists('last_{}.pth'.format(model_name)):
    print('load_pretrained')
    training_state = torch.load('last_{}.pth'.format(model_name), map_location=torch.device("cpu"))
    start_epoch = training_state['epoch'] + 1
    new_weight = net.state_dict()
    new_weight.update(training_state['model_state'])
    net.load_state_dict(new_weight)

net.to(device)
num_gpus = torch.cuda.device_count()

net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank],
                                          output_device=args.local_rank)
# Traning loader
Train_set = Train_Loader(input_path, gt_path, crop_size)
train_sampler = DistributedSampler(Train_set)
dataloader_train = DataLoader(Train_set, sampler=train_sampler, batch_size=batch_size // num_gpus,
                              num_workers=1, pin_memory=True, find_unused_parameters=False)  # find_unused_parameters 建議不要開可以debug用

# Model and optimizer
optimizer = optim.Adam(net.parameters(), lr=init_lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=end_epoch, eta_min=min_lr)

if os.path.exists('last_{}.pth'.format(model_name)):
    new_optimizer = optimizer.state_dict()
    new_optimizer.update(training_state['optimizer_state'])
    optimizer.load_state_dict(new_optimizer)
    new_scheduler = scheduler.state_dict()
    new_scheduler.update(training_state['scheduler_state'])
    scheduler.load_state_dict(new_scheduler)

def calc_psnr(result, gt):
    result = result.cpu().numpy()
    gt = gt.cpu().numpy()
    mse = np.mean(np.power((result - gt), 2))
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

best_psnr = 0
best_epoch = 0
for epoch in range(start_epoch, end_epoch):
    train_sampler.set_epoch(epoch)
    tq = tqdm.tqdm(dataloader_train, total=len(dataloader_train))
    tq.set_description(
        'Epoch {}, lr {:.8f}'.format(epoch, optimizer.param_groups[0]['lr']))

    total_train_loss = 0.
    total_train_psnr = 0.
    for idx, sample in enumerate(tq):

        input, gt = sample['input'].to(device), sample['gt'].to(device)

        optimizer.zero_grad()
        output = net(input).clamp(0, 1)
        loss = L1_Loss(output, gt)
        loss.backward()
        optimizer.step()

        psnr = calc_psnr(output.detach(), gt.detach())
        total_train_loss += loss.item()
        total_train_psnr += psnr

        tq.set_postfix(PSNR =(total_train_psnr / (idx + 1)),
                       Loss=(total_train_loss / (idx + 1)))


    scheduler.step()

    writer.add_scalar('Train_loss', total_train_loss / (idx + 1), epoch)
    writer.add_scalar('Train_PSNR', total_train_psnr / (idx + 1), epoch)

    # save parameters
    scheduler_state = scheduler.state_dict()
    optimizer_state = optimizer.state_dict()
    net_state = net.module.state_dict()
    training_state = {'epoch': epoch, 'model_state': net_state,
                      'scheduler_state': scheduler_state, 'optimizer_state': optimizer_state}

    torch.save(training_state, 'last_{}.pth'.format(model_name))

    if (epoch % check_point_epoch) == 0:
        torch.save(training_state, 'epoch_{}_{}.pth'.format(epoch, model_name))

    if epoch == (end_epoch - 1):
        torch.save(net_state, 'final_{}.pth'.format(model_name))

