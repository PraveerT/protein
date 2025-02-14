from torch import nn
from torch.nn import functional as F
import torch

class MLPBlock(nn.Module):
    def __init__(self, out_channel, dimension, with_bn=True):
        super(MLPBlock, self).__init__()
        self.layer_list = []
        if dimension == 1:
            for idx, channels in enumerate(out_channel[:-1]):
                if with_bn:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv1d(channels, out_channel[idx + 1], kernel_size=1),
                            nn.BatchNorm1d(out_channel[idx]),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv1d(channels, out_channel[idx + 1], kernel_size=1),
                        )
                    )
        elif dimension == 2:
            for idx, channels in enumerate(out_channel[:-1]):
                if with_bn:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv2d(channels, out_channel[idx + 1], kernel_size=(1, 1)),
                            nn.BatchNorm2d(out_channel[idx + 1]),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv2d(channels, out_channel[idx + 1], kernel_size=(1, 1)),
                        )
                    )
        self.layer_list = nn.ModuleList(self.layer_list)

    def forward(self, output):
        for layer in self.layer_list:
            output = layer(output)
        return output


if __name__=="__main__":
    inputs = torch.rand((2, 512, 32, 64))
    mlpblock = MLPBlock([512, 1024], 2)
    output = mlpblock(inputs)
    print ("success")
