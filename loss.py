import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss

class L1_Loss(nn.Module):
    def __init__(self, ):
        super(L1_Loss, self).__init__()

    def forward(self, x, y):
        loss = nn.L1Loss()(x, y)
        return loss

class L2_Loss(nn.Module):
    def __init__(self, ):
        super(L2_Loss, self).__init__()

    def forward(self, x, y):
        loss = nn.MSELoss()(x, y)
        return loss

class FFT_Loss(nn.Module):
    def __init__(self, ):
        super(FFT_Loss, self).__init__()

    def forward(self, x, y):
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft = torch.stack((x_fft.real, x_fft.imag), -1)
        y_fft = torch.fft.fft2(y, dim=(-2, -1))
        y_fft = torch.stack((y_fft.real, y_fft.imag), -1)
        loss = nn.L1Loss()(x_fft, y_fft)
        return loss