from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class CustomConv2d(nn.Module):
    """Custom 2D Convolution that enables circular padding along the width dimension only"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,  # Left, Right, Up, Down | L-R are circular
    ):
        """Init custom 2D Conv with circular padding"""
        super().__init__()
        self.padding = padding

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, padding='valid')

    def forward(self, x):
        """Forward custom 3D convolution
        Args:
            x (torch.tensor): Input tensor
        Returns:
            torch.tensor: Output tensor
        """

        # default NCHW
        # (1,1) for circular width, (1,1) for same height
        # so (1,1,1,1) altogether

        x = F.pad(x, (self.padding[0], self.padding[1], 0, 0), mode="circular")
        x = F.pad(x, (0, 0, self.padding[2], self.padding[3]), mode="constant")
        x = self.conv(x)
        return x


class MMCNN(nn.Module):
    def __init__(self, img_height=16, img_width=200):
        super(MMCNN, self).__init__()

        self.im_conv1 = CustomConv2d(2, 4, 5, (2, 2))
        self.im_pool1 = nn.MaxPool2d(2)
        self.im_conv2 = CustomConv2d(4, 8, 3, (2, 2))
        self.im_pool2 = nn.MaxPool2d(4)
        self.im_conv3 = CustomConv2d(8, 8, 3, (2, 2))
        self.im_pool3 = nn.MaxPool2d(4)
        self.im_conv4 = CustomConv2d(8, 16, 3, (2, 2))
        self.im_pool4 = nn.MaxPool2d(4)
        self.im_conv5 = CustomConv2d(16, 32, 3, (2, 2))
        self.im_pool4 = nn.MaxPool2d(8)

        # Image part
        # self.im_conv1 = nn.Conv2d(2, 4, 5)
        # self.im_conv2 = nn.Conv2d(4, 4, 3)
        # self.im_conv3 = nn.Conv2d(4, 8, 3)
        # self.im_conv4 = nn.Conv2d(8, 8, 3, padding='same')
        # self.im_conv5 = nn.Conv2d(8, 8, 3, padding='same')
        # self.im_pool = nn.MaxPool2d(2)
        # self.im_fc1 = nn.Linear(32*3*95, 100)s

        # Scalar part
        self.sc_fc1 = nn.Linear(3, 100)
        # self.sc_fc2 = nn.Linear(100, 100)
        # self.sc_fc3 = nn.Linear(100, 100)

        # Concatenation part
        self.con_fc1 = nn.Linear(228, 128)
        self.con_fc2 = nn.Linear(128, 64)
        self.con_fc3 = nn.Linear(64, 32)
        self.con_fc4 = nn.Linear(32, 16)
        self.con_fc5 = nn.Linear(16, 8)
        self.con_fc6 = nn.Linear(8, 1)

        # self.con_fc1 = nn.Linear(476, 100)
        # self.con_fc2= nn.Linear(100, 64)
        # self.con_fc3 = nn.Linear(64, 32)
        # self.con_fc4 = nn.Linear(32, 16)
        # self.con_fc5 = nn.Linear(16, 1)
        self.save_cnn_embedding = False
        self.cnn_embedding = None

    def forward(self, im_inp, sc_inp):
        '''
        Args:
            im_inp: image input
            sc_inp: scalar input
        '''
        if self.cnn_embedding is None:
            im_inp = F.relu(self.im_conv1(im_inp))
            im_inp = self.im_pool1(im_inp)
            im_inp = F.relu(self.im_conv2(im_inp))
            im_inp = self.im_pool2(im_inp)
            im_inp = F.relu(self.im_conv3(im_inp))
            im_inp = self.im_pool3(im_inp)
            im_inp = F.relu(self.im_conv4(im_inp))
            im_inp = self.im_pool4(im_inp)
            # print(im_inp.shape)
            im_inp = torch.flatten(im_inp, 1)
            # im_inp = F.relu(self.im_fc1(im_inp))
        if self.save_cnn_embedding:
            if self.cnn_embedding is None:
                self.cnn_embedding = im_inp
            else:
                im_inp = self.cnn_embedding

        sc_inp = F.relu(self.sc_fc1(sc_inp))
        # sc_inp = F.relu(self.sc_fc2(sc_inp))
        # sc_inp = F.relu(self.sc_fc3(sc_inp))
        sc_inp = torch.flatten(sc_inp, 1)  # This might not be needed

        con_inp = torch.cat((im_inp, sc_inp), 1)
        con_inp = F.relu(self.con_fc1(con_inp))
        con_inp = F.relu(self.con_fc2(con_inp))
        con_inp = F.relu(self.con_fc3(con_inp))
        con_inp = F.relu(self.con_fc4(con_inp))
        con_inp = F.relu(self.con_fc5(con_inp))
        con_inp = self.con_fc6(con_inp)
        # con_inp = F.relu(self.con_fc3(con_inp))
        # con_inp = self.con_fc4(con_inp)

        return con_inp


#########################################################################

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, bn_d=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0], momentum=bn_d)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1], momentum=bn_d)
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


# number of layers per model
model_blocks = {
    21: [1, 1, 2, 2, 1],
    53: [1, 2, 8, 8, 4],
}


class Darknet(nn.Module):
    """
       Class for DarknetSeg. Subclasses PyTorch's own "nn" module
    """

    def __init__(self):
        super(Darknet, self).__init__()
        self.use_range = True
        self.use_xyz = False
        self.use_remission = False
        self.drop_prob = 0.01
        self.bn_d = 0.01
        self.layers = 21
        print("Using DarknetNet" + str(self.layers) + " Backbone")

        # input depth calc
        self.input_depth = 0
        self.input_idxs = []
        if self.use_range:
            self.input_depth += 1
            self.input_idxs.append(0)
        if self.use_xyz:
            self.input_depth += 3
            self.input_idxs.extend([1, 2, 3])
        if self.use_remission:
            self.input_depth += 1
            self.input_idxs.append(4)
        print("Depth of backbone input = ", self.input_depth)

        # stride play
        self.strides = [2, 2, 2, 2, 2]

        # check that darknet exists
        assert self.layers in model_blocks.keys()

        # generate layers depending on darknet type
        self.blocks = model_blocks[self.layers]

        # input layer
        self.conv1 = nn.Conv2d(self.input_depth, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=self.bn_d)
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv1x1 = nn.Conv2d(
            512, 1, kernel_size=3, stride=1, padding=1, bias=False)

        # encoder
        self.enc1 = self._make_enc_layer(BasicBlock, [32, 32], self.blocks[0],
                                         stride=self.strides[0], bn_d=self.bn_d)
        self.enc2 = self._make_enc_layer(BasicBlock, [32, 64], self.blocks[1],
                                         stride=self.strides[1], bn_d=self.bn_d)
        self.enc3 = self._make_enc_layer(BasicBlock, [128, 256], self.blocks[2],
                                         stride=self.strides[2], bn_d=self.bn_d)
        self.enc4 = self._make_enc_layer(BasicBlock, [256, 512], self.blocks[3],
                                         stride=self.strides[3], bn_d=self.bn_d)
        self.enc5 = self._make_enc_layer(BasicBlock, [512, 1024], self.blocks[4],
                                         stride=self.strides[4], bn_d=self.bn_d)
        self.pool1 = nn.MaxPool2d(1)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(3)

        # concat part
        self.fc1 = nn.Linear(7168, 1024)  # 4096
        self.fc2 = nn.Linear(1024, 100)
        self.fc3 = nn.Linear(100, 1)

        # for a bit of fun
        self.dropout = nn.Dropout2d(self.drop_prob)

        # last channels
        self.last_channels = 1024

    # make layer useful function
    def _make_enc_layer(self, block, planes, blocks, stride, bn_d=0.1):
        layers = []

        #  downsample
        layers.append(("conv", nn.Conv2d(planes[0], planes[1],
                                         kernel_size=3,
                                         stride=[1, stride], dilation=1,
                                         padding=1, bias=False)))
        layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
        layers.append(("relu", nn.LeakyReLU(0.1)))

        #  blocks
        inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i),
                           block(inplanes, planes, bn_d)))

        return nn.Sequential(OrderedDict(layers))

    def run_layer(self, x, layer):
        return layer(x)

    def forward(self, x):
        # filter input
        x = x[:, self.input_idxs]

        # run cnn
        # first layer
        x = self.run_layer(x, self.conv1)
        x = self.run_layer(x, self.bn1)
        x = self.run_layer(x, self.relu1)

        # all encoder blocks with intermediate dropouts
        x = self.run_layer(x, self.enc1)
        x = self.pool2(x)
        x = self.run_layer(x, self.dropout)
        x = self.pool2(x)
        x = self.run_layer(x, self.enc2)
        x = self.run_layer(x, self.dropout)
        x = self.pool2(x)
        # x = self.run_layer(x, self.enc3)
        # x = self.run_layer(x, self.dropout)
        # x = self.pool2(x)
        # x = self.run_layer(x, self.enc4)
        # x = self.run_layer(x, self.dropout)
        # x = self.pool2(x)
        # x = self.run_layer(x, self.enc5)
        # x = self.run_layer(x, self.dropout)
        # x = self.pool3(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def get_last_depth(self):
        return self.last_channels

    def get_input_depth(self):
        return self.input_depth


class LiteNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.drop_prob = 0.01
        self.bn_d = 0.01
        print('Using LiteNet backnone!')

        # Conv layers
        self.conv1 = CustomConv2d(1, 4, (1, 2), (1, 0, 0, 0))
        self.conv2 = CustomConv2d(4, 8, (1, 2), (1, 0, 0, 0))
        self.conv3 = CustomConv2d(8, 8, (3, 3), (0, 0, 1, 1))
        self.conv4 = CustomConv2d(8, 4, (1, 1), (0, 0, 0, 0))
        self.conv5 = CustomConv2d(4, 1, (1, 1), (0, 0, 0, 0))

        # Batch Norms
        self.bn1 = nn.BatchNorm2d(4, momentum=self.bn_d)
        self.bn2 = nn.BatchNorm2d(8, momentum=self.bn_d)
        self.bn3 = nn.BatchNorm2d(8, momentum=self.bn_d)
        self.bn4 = nn.BatchNorm2d(4, momentum=self.bn_d)
        self.bn5 = nn.BatchNorm2d(1, momentum=self.bn_d)

        # Activations
        self.relu = nn.LeakyReLU(0.1)

        # Pooling
        self.pool1 = nn.MaxPool2d((1, 2))

        # FC
        self.fc1 = nn.Linear(14384, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):

        # Convs
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.pool1(x)

        x = torch.flatten(x, 1)

        # FCs
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    # # data pts, channels, height, width
    # im_inp = torch.randn(16, 2, 16, 1800)
    # sc_inp = torch.randn(16, 3)
    # net = MMCNN()
    # out = net(im_inp, sc_inp)
    # print(out.shape)
    # # breakpoint()

    # # data pts, channels, height, width
    # im_inp = torch.randn(16, 1, 16, 1800)
    # net = Darknet()
    # out = net(im_inp)
    # print(out.shape)

    # data pts, channels, height, width
    im_inp = torch.randn(16, 1, 16, 1800)
    net = LiteNet()
    out = net(im_inp)
    print(out.shape)
    breakpoint()
