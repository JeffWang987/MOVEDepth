import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from movedepth.layers import ConvBlock, Conv3x3, upsample


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, match_conv=False, ddv=False, discret='UD',
                mono_conf=False, mono_bins=128):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.ddv = ddv
        self.mono_conf = mono_conf
        self.discret = discret
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.mono_bins = mono_bins


        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            if self.ddv:
                self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.mono_bins)    
            else:
                self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)        
            if self.mono_conf:
                self.convs[("confconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)        

        if match_conv:
            self.convs[("matchconv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_ch_dec[0])

        self.match_conv = match_conv
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def get_depth(self, depth_labels=128):
        if self.discret == 'SID':
            alpha_ = torch.FloatTensor([0.001])
            beta_ = torch.FloatTensor([1])
            t = []
            for K in range(depth_labels):
                K_ = torch.FloatTensor([K])
                t.append(torch.exp(torch.log(alpha_) + torch.log(beta_ / alpha_) * K_ / depth_labels))
            t = torch.FloatTensor(t).unsqueeze(0).unsqueeze(2).unsqueeze(3)  # 1 D 1 1

        elif self.discret == 'UD':
            t = torch.linspace(0.001, 1, depth_labels, requires_grad=False).unsqueeze(0).unsqueeze(2).unsqueeze(3)  # 1 D 1 1
        return t  # 1 D 1 1


    def forward(self, input_features, no_disp=False, no_match=True, outs=0):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1+outs, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                if not no_disp:
                    if self.ddv:
                        disp_feat = self.convs[("dispconv", i)](x)  # B D H W
                        depth_grid = self.get_depth(self.mono_bins)
                        depth_grid = depth_grid.repeat(disp_feat.shape[0],1,disp_feat.shape[2],disp_feat.shape[3]).to(disp_feat.device).float()  # B 128 H W
                        self.outputs[("ddv", i)] = F.softmax(disp_feat, dim=1)  # B D H W
                        self.outputs[("disp", i)] = (self.outputs[("ddv", i)] * depth_grid).sum(1, keepdim=True)  # B 1 H W
                    else:
                        self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                        if self.mono_conf:
                            self.outputs[("mono_conf", i)] = F.elu(self.convs[("confconv", i)](x)) + 1.0 + 1e-10
                            
            if i==0 and not no_match and self.match_conv:
                self.outputs[("match", 0)] = self.convs[("matchconv", 0)](x)

        return self.outputs



class MPMDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, num_bins=8):
        super(MPMDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = [scales]
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])


        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in [2,1,0]:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)        

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.reduce_conv = nn.Conv2d(self.num_ch_dec[2]+num_bins, self.num_ch_dec[2], kernel_size=1, stride=1, padding=0)



    def forward(self, costvol, mono_feat):
        self.outputs = {}

        # decoder
        x = mono_feat[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            if i == 2:
                x = upsample(x)
                x = [self.reduce_conv(torch.cat([x, costvol], dim=1))]  # B C H W
            else:
                x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [mono_feat[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            if i in [2, 1, 0]:
                self.outputs[("mpm_disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs

class Conv3DBlock(nn.Module):

    def __init__(self, in_channels, base_channels, kernel_size=3, stride=1, pad=1):
        super(Conv3DBlock, self).__init__()

        self.conv = nn.Conv3d(in_channels, base_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.convout = nn.Conv3d(base_channels, 1, kernel_size, stride=stride, padding=pad, bias=False)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        out = self.convout(out)
        return out

class DepthDecoder3D(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), discret='UD', mono_bins=96, min_d=0.1, max_d=10.0):
        super(DepthDecoder3D, self).__init__()

        self.upsample_mode = 'nearest'
        self.scales = scales
        self.discret = discret
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # update decoder
        self.convs = OrderedDict()
        self.channel_rec = {}
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if  i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            self.channel_rec['{}'.format(i)] = num_ch_out
        
        # 3d depth cnn
        self.mono_bins = mono_bins
        self.reg_c = 4
        for i in self.scales:
            self.convs[("depth_expand", i)] = ConvBlock(self.channel_rec['{}'.format(i)], self.mono_bins*self.reg_c)
            self.convs[("depth_3dcnn", i)] = Conv3DBlock(self.reg_c, self.reg_c)


        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.min_d = min_d
        self.max_d = max_d
        self.depth_grid = self.get_depth(self.mono_bins, self.min_d, self.max_d)

    def get_depth(self, depth_labels, min_d, max_d):
        if self.discret == 'SID':
            alpha_ = torch.FloatTensor([min_d])
            beta_ = torch.FloatTensor([max_d])
            t = []
            for K in range(depth_labels):
                K_ = torch.FloatTensor([K])
                t.append(torch.exp(torch.log(alpha_) + torch.log(beta_ / alpha_) * K_ / depth_labels))
            t = torch.FloatTensor(t).unsqueeze(0).unsqueeze(2).unsqueeze(3)  # 1 D 1 1

        elif self.discret == 'UD':
            t = torch.linspace(min_d, max_d, depth_labels, requires_grad=False).unsqueeze(0).unsqueeze(2).unsqueeze(3)  # 1 D 1 1
        return t.cuda()  # 1 D 1 1


    def forward(self, input_features, min_d=None, max_d=None):
        if min_d is not None:
            self.depth_grid = self.get_depth(self.mono_bins, min_d, max_d)
        self.outputs = {}
        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            # udpdate next stage hidden state
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            # calculate this stage depth
            if i in self.scales:
                depth_feat =  self.convs[("depth_expand", i)](x)
                B, _, H, W = depth_feat.shape
                depth_feat = depth_feat.reshape(B, self.reg_c, self.mono_bins, H, W)  # B C D H W
                depth_prob = torch.softmax(self.convs[("depth_3dcnn", i)](depth_feat), dim=1)[:, 0]  # B D H W
                self.outputs[("mono_depth", i)] = (depth_prob * self.depth_grid).sum(dim=1, keepdim=True)  # B 1 H W



        return self.outputs


class DepthDecoderbin(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, mono_bins=96):
        super(DepthDecoderbin, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.mono_bins = mono_bins

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            self.convs[("binconv", s)] = Conv3x3(self.num_ch_dec[s], self.mono_bins)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                self.outputs[("bin", i)] = torch.softmax(self.convs[("binconv", i)](x), dim=1)  # B N H W

        return self.outputs



class DepthDecoder3head(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder3head, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels*4)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                out_disp = self.sigmoid(self.convs[("dispconv", i)](x))
                self.outputs[("disp_rough", i)] = out_disp[:,0:1]
                self.outputs[("disp_1", i)] = out_disp[:,1:2]
                self.outputs[("disp_2", i)] = out_disp[:,2:3]
                self.outputs[("disp_3", i)] = out_disp[:,3:]

        return self.outputs



class UncertNet(nn.Module):
    
    def __init__(self):
        super(UncertNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.head_convs = nn.Conv2d(8, 1, 3, 1, 1, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += x
        out = self.head_convs(out)
        out = torch.sigmoid(out)
        return out  # B 1 H W

