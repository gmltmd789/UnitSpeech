""" from https://github.com/jaywalnut310/glow-tts """
import torch
import torch.nn as nn
import torch.nn.functional as F

from unitspeech.base import BaseModule


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1).contiguous()
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1).contiguous()


class DurationPredictor(BaseModule):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, spk_emb_dim=0):
        super(DurationPredictor, self).__init__()
        in_channels = in_channels + spk_emb_dim

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout

        self.drop = torch.nn.Dropout(p_dropout)
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels,
                                      kernel_size, padding=kernel_size//2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = torch.nn.Conv1d(filter_channels, filter_channels,
                                      kernel_size, padding=kernel_size//2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = torch.nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False):
        x = torch.detach(x)
        if g is not None:
            x = torch.cat([x, g.transpose(1, 2).repeat(1, 1, x.shape[-1])], dim=1)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        logw = self.proj(x * x_mask) * x_mask
        if not reverse:
            logw_ = torch.log(w + 1e-6) * x_mask
            return torch.sum((logw - logw_) ** 2) / torch.sum(x_mask)  # for averaging
        else:
            return logw