import math
import torch
import numpy as np
import os
import time
from torch.nn import functional as F
import torchvision
import cv2
import glob
from scipy.ndimage import gaussian_filter

from model.baseline_model import Baseline_Model

input_path = './dataset/NH-HAZE/test/haze'
gt_path = './dataset/NH-HAZE/test/gt'
out_path = './out/final_Baseline_Model'
model_name = './final_Baseline_Model.pth'
if not os.path.isdir(out_path):
        os.mkdir(out_path)

net = Baseline_Model().cuda()
net.load_state_dict(torch.load(model_name))
net.cuda()

input_name_path = sorted(glob.glob(os.path.join(input_path, "*")))
gt_name_path = sorted(glob.glob(os.path.join(gt_path, "*")))

def calc_psnr(result, gt):
    result = result.cpu().numpy()
    gt = gt.cpu().numpy()
    mse = np.mean(np.power((result - gt), 2))
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def calc_ssim(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):
    # Processing input image
    img1 = np.array(img1, dtype=np.float32)
    img1 = img1.transpose((2, 0, 1))

    # Processing gt image
    img2 = np.array(img2, dtype=np.float32)
    img2 = img2.transpose((2, 0, 1))

    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))

    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_num / ssim_den
    return np.mean(ssim_map)


total_psnr = 0
total_time = 0
count = 0
for i in range(len(input_name_path)):
    input = cv2.imread(input_name_path[i]).astype(np.float32) / 255
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    input = torch.from_numpy(input.transpose((2, 0, 1))).unsqueeze(0).cuda()

    gt = cv2.imread(gt_name_path[i]).astype(np.float32) / 255
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    gt = torch.from_numpy(gt.transpose((2, 0, 1))).unsqueeze(0).cuda()

    name = input_name_path[i].split('/')[-1]
    save_path = os.path.join(out_path, name)

    with torch.no_grad():
        torch.cuda.synchronize()
        start_time = time.time()
        output = net(input).clamp(0, 1)
        torch.cuda.synchronize()
        end_time = time.time()
        torchvision.utils.save_image(output, save_path)

    psnr = calc_psnr(output.detach(), gt.detach())
    total_psnr += psnr
    inference_time = end_time - start_time
    total_time += inference_time
    print(name, total_psnr / (i+1), total_time / (i+1))

print(model_name)