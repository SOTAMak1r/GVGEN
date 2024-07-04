
import torch
import torch.nn as nn


class CLIPImageEncoderAdapter(nn.Module):
    def __init__(self, c_in=768, reduction=4, ratio=0.6):
        super(CLIPImageEncoderAdapter, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            # nn.ReLU(inplace=True)
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(c_in,              c_in // reduction),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(c_in // reduction, c_in // reduction),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(c_in // reduction, c_in),
        # )
        self.ratio = ratio

    def forward(self, x):
        x_ = self.fc(x)
        feature = self.ratio * x_ + (1 - self.ratio) * x # residual connect
        return feature

