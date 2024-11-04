import torch
import torch.nn as nn


class Avg_THW(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Avg_THW, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Linear(in_channel, out_channel, bias=False)
        self.ln1 = nn.LayerNorm(out_channel)
        self.sigmoid = nn.Sigmoid()

        nn.init.normal_(self.conv.weight, 0, 0.001)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        y = self.avg_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)
        y = self.conv(y)
        y = self.ln1(y)
        y = self.sigmoid(y).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = x * y.expand_as(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x


class Avg_T(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Avg_T, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)
        y = y.permute(0, 2, 1).contiguous()
        y = self.conv(y)
        y = self.bn1(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).contiguous()
        y = y.unsqueeze(-1).unsqueeze(-1)
        x = x * y.expand_as(x)

        return x


class Avg_HW(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Avg_HW, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

    def forward(self, x):
        y = x.mean(dim=1)
        y = self.conv(y)
        y = self.bn1(y)
        y = self.sigmoid(y).unsqueeze(1)
        x = x * y.expand_as(x)

        return x


class Att_C(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Att_C, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, in_channel * 4, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(in_channel * 4)

        self.conv2 = nn.Conv1d(in_channel * 4, out_channel, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(in_channel)

    def forward(self, x):
        input = x
        bsz, t, c, w, h = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().reshape(bsz, c, -1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x.reshape(bsz, c, t, w, h).permute(0, 2, 1, 3, 4).contiguous()
        x = input + x
        return x


class Attention(nn.Module):
    def __init__(self, channels):
        super(Attention, self).__init__()
        # x.shape[2] // 6
        self.ratio = channels
        self.avg_thw = Avg_THW(channels, channels)
        self.avg_t = Avg_T(channels, channels)
        self.avg_hw = Avg_HW(channels, channels)
        self.att_c = Att_C(channels, channels)

    def forward(self, x):
        new_x = torch.zeros_like(x)
        new_x[:, :, :self.ratio] = self.avg_t(x[:, :, :self.ratio])
        new_x[:, :, self.ratio:self.ratio * 2] = self.avg_thw(x[:, :, self.ratio:self.ratio * 2])
        new_x[:, :, self.ratio * 2:self.ratio * 3] = self.avg_hw(x[:, :, self.ratio * 2:self.ratio * 3])
        new_x[:, :, self.ratio * 3:self.ratio * 4] = self.att_c(x[:, :, self.ratio * 3:self.ratio * 4])
        new_x[:, :, self.ratio * 4:] = x[:, :, self.ratio * 4:]
        new_x = new_x.mean(1)
        return new_x


if __name__ == '__main__':
    x = torch.randn(1, 2, 12, 320, 480)  # (5,4,12,96,96)  (5, 2, 12, 320, 480)
    model = Attention(x.shape[2] // 6)
    from torchinfo import summary

    summary(model, input_size=(1, 4, 12, 96, 96))
