import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.unet import SimpleUNet

class PointBlock(nn.Module):
    '''
    input_channel: 输入通道
    base_channel: 中间层通道基数
    output_channel: base_channel * 4
    '''
    def __init__(self, input_channel: int = 4, base_channel=16):
        super(PointBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channel, base_channel, 1)
        self.conv2 = torch.nn.Conv2d(base_channel, base_channel * 2, 1)
        self.conv3 = torch.nn.Conv2d(base_channel * 2, base_channel * 4, 1)
        self.bn1 = nn.BatchNorm2d(base_channel)
        self.bn2 = nn.BatchNorm2d(base_channel * 2)
        self.bn3 = nn.BatchNorm2d(base_channel * 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

class ResBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        padding: int = 1,
        kernel_size: int = 3,
        stride=1,
    ):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.BatchNorm2d(n_outputs)

        self.conv2 = nn.Conv2d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=1,
            padding=padding,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.BatchNorm2d(n_outputs)
        self.net = nn.Sequential(
            self.conv1,
            self.dropout1,
            self.relu1,
            self.conv2,
            self.dropout2,
            self.relu2,
        )

        downsample_kernel = (1, 1)
        self.shortcut = nn.Sequential()
        if stride != 1:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    n_inputs, n_outputs, kernel_size=downsample_kernel, stride=stride, bias=False
                ),
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + self.shortcut(x))

point_base_channel = 16
class LaneModelCls(nn.Module):
    def __init__(
        self,
        max_points_in_grid: int,
        grid_height: int,
        grid_width: int,
        input_channel: int = 4,
        multi_maxpool: bool = False,
    ):
        super(LaneModelCls, self).__init__()

        self.grid_height = grid_height
        self.grid_width = grid_width

        self.point_feature_module = PointBlock(
            input_channel=input_channel, base_channel=point_base_channel
        )
        self.max_pool = nn.MaxPool2d((1, max_points_in_grid))

        # layers = []
        # if multi_maxpool:
        #     for _ in range(6):
        #         layers.append(nn.MaxPool2d((1, 2), stride=(1, 2)))
        #     self.max_pool = nn.Sequential(*layers)
        # else:
        #     self.max_pool = nn.MaxPool2d((1, max_points_in_grid))

        conv_base_channel = point_base_channel * 4
        self.conv1 = nn.Conv2d(
            in_channels=conv_base_channel,
            out_channels=conv_base_channel,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        self.bn1 = nn.BatchNorm2d(conv_base_channel)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = ResBlock(
            n_inputs=conv_base_channel, n_outputs=conv_base_channel * 2, stride=2
        )
        self.layer2 = ResBlock(
            n_inputs=conv_base_channel * 2, n_outputs=conv_base_channel * 4, stride=2
        )
        self.layer3 = ResBlock(
            n_inputs=conv_base_channel * 4, n_outputs=conv_base_channel * 8, stride=2
        )

        # self.max_pool2 = nn.MaxPool2d((3, 2))
        # self.max_pool2_1 = nn.MaxPool2d((2, 2), padding=(1, 0))
        # self.max_pool2_2 = nn.MaxPool2d((2, 1))
        
        self.max_pool2 = nn.AdaptiveAvgPool2d(1)

        self.linear1 = nn.Linear(conv_base_channel * 8, 2)

    def forward(self, x):
        # x: [bs, 4, M*N, dim]
        output = {}
        x = self.point_feature_module(x) # [bs, 64, M*N, dim]
        x = self.max_pool(x) # [bs, 64, M*N, 1]
        x = x.view(x.shape[0], x.shape[1], self.grid_height, self.grid_width) # [bs, 64, M, N]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # x = self.max_pool2_1(x)
        # x = self.max_pool2_2(x)
        x = self.max_pool2(x)

        x = x.view(x.shape[0], -1)
        x = self.linear1(x)

        output['pred'] = x
        return output

class LaneModelSeg(nn.Module):
    def __init__(
        self,
        max_points_in_grid: int,
        grid_height: int,
        grid_width: int,
        input_channel: int = 4
    ):
        super(LaneModelSeg, self).__init__()

        self.grid_height = grid_height
        self.grid_width = grid_width

        self.point_feature_module = PointBlock(
            input_channel=input_channel, base_channel=point_base_channel
        )
        self.max_pool = nn.MaxPool2d((1, max_points_in_grid))
        
        self.unet = SimpleUNet(in_ch=point_base_channel * 4, out_ch=2, base_chn=64)
        
    def forward(self, x):
        '''
        x: [bs, 4, M*N, dim]
        '''
        output = {}
        x = self.point_feature_module(x) # [bs, 64, M*N, dim]
        # print(x.shape)
        x = self.max_pool(x) # [bs, 64, M*N, 1]
        # print(x.shape)
        # print(self.grid_height, self.grid_width)
        x = x.view(x.shape[0], x.shape[1], self.grid_height, self.grid_width) # [bs, 64, M, N]

        x = self.unet(x)
        
        output['pred'] = x

        return output


if __name__ == '__main__':
    pass