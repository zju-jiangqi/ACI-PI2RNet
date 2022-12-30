from tokenize import group
from cv2 import groupRectangles
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


#D2Conv
class DynamicDeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=True):
        
        super(DynamicDeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)

        self.f_conv = nn.Conv2d(inc, outc*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride) #计算位移前的特征提取
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, feature_size, feature_context, x):
        offset = self.p_conv(feature_size)
        psffilter = self.f_conv(feature_context)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(feature_size))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        
        out = FAC(x_offset, psffilter, kernel_size=self.kernel_size, stride=self.kernel_size)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset


class PI2RNet(nn.Module):

    def __init__(self, in_chn=3, psf_schn=3, wf=64, depth=3, relu_slope=0.2, hin_position_left=0, hin_position_right=3):
        super(PI2RNet, self).__init__()
        self.depth = depth
        self.down_path_1 = nn.ModuleList() #fRN
        self.down_path_2 = nn.ModuleList() #H*
        self.conv_01 = nn.Conv2d((in_chn+in_chn), wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d((in_chn+psf_schn), wf, 3, 1, 1)
        

        
        prev_channels = self.get_input_chn(wf)
        prev_channels_half = prev_channels
        for i in range(depth): #0,1,2,3,4
            use_HIN = True if hin_position_left <= i and i <= hin_position_right else False
            downsample = True if (i+1) < depth else False
            self.down_path_1.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_HIN=use_HIN))
            self.down_path_2.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_HIN=False))
            prev_channels = (2**i) * wf
            prev_channels_half = (2**(i-1)) * wf

        self.psffeature = ConvFeature(prev_channels_half, relu_slope)
        self.ddconv1 = DynamicDeformConv2d(prev_channels, prev_channels, kernel_size=3, modulation=True)

        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList() 
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.up_path_2.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.skip_conv_1.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            self.skip_conv_2.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            prev_channels = (2**i)*wf
        
        self.convfeature = ConvFeature(32, relu_slope)
        self.kernelconv = DynamicDeformConv2d(wf, in_chn, kernel_size=5, modulation=True)
        self.aspp = ContextBlock(input_channel=256, output_channel=256, square=True)


        self.last = conv3x3(prev_channels, in_chn, bias=True)
        
        

    def forward(self, x, psf):
        image = x
        xhat = torch.cat([x,psf], dim=1)
        #stage 1
        
        x1hat = self.conv_02(xhat)
        encs = []
        encspsf = []


        for i, down in enumerate(self.down_path_2):
            if (i+1) < self.depth:
                x1hat, xxx = down(x1hat)
                encspsf.append(xxx)
            else:
                x1hat = down(x1hat)   

        psfer = x1hat

        for i, up in enumerate(self.up_path_2):
            x1hat = up(x1hat, self.skip_conv_2[i](encspsf[-i-1]))


        feature_one = x1hat
        x1hat1, x1hat2 = self.convfeature(x1hat)
        image_retore1 = self.kernelconv(x1hat1, x1hat2, image)
        x1 = self.conv_01(torch.cat([image, image_retore1],dim=1)) + feature_one

        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:
                x1, x1_up = down(x1)
                encs.append(x1_up)
            else:
                x1 = down(x1)

        size_feature, context_feature = self.psffeature(psfer)
        x1 = self.ddconv1(size_feature, context_feature, x1)
        x1 = self.aspp(x1)
        
        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i-1])) #后面的是bridge 跳跃连接

        out = self.last(x1) + image

        return out, image_retore1

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)  



class ContextBlock(nn.Module):
    def __init__(self, input_channel=32, output_channel=32, square=False):
        super().__init__()
        self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        if square:
            self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1, 1)
            self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 2, 2)
            self.conv3 = nn.Conv2d(output_channel, output_channel, 3, 1, 4, 4)
            self.conv4 = nn.Conv2d(output_channel, output_channel, 3, 1, 8, 8)
        else:
            self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1, 1)
            self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 2, 2)
            self.conv3 = nn.Conv2d(output_channel, output_channel, 3, 1, 3, 3)
            self.conv4 = nn.Conv2d(output_channel, output_channel, 3, 1, 4, 4)
        self.fusion = nn.Conv2d(4*output_channel, input_channel, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x):
        x_reduce = self.conv0(x)
        conv1 = self.lrelu(self.conv1(x_reduce))
        conv2 = self.lrelu(self.conv2(x_reduce))
        conv3 = self.lrelu(self.conv3(x_reduce))
        conv4 = self.lrelu(self.conv4(x_reduce))
        out = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.fusion(out) + x
        return out
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()               
                      
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_HIN=False):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size//2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)

        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out

class Subspace(nn.Module):

    def __init__(self, in_size, out_size):
        super(Subspace, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(UNetConvBlock(in_size, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x + sc

class ConvFeature(nn.Module):
    def __init__(self, in_size, relu_slope):
        super(ConvFeature, self).__init__()
        self.conv_size_01 = nn.Conv2d(in_size, 2*in_size, kernel_size=1, stride = 1, padding=0, bias=True)
        self.conv_size_02 = nn.Conv2d(2*in_size, 2*in_size, kernel_size=3, stride = 1, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_context_01 = nn.Conv2d(in_size, 2*in_size, kernel_size=1, stride = 1, padding=0, bias=True)
        self.conv_context_02 = nn.Conv2d(2*in_size, 2*in_size, kernel_size=3, stride = 1, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        x_0, x_1 = x.chunk(2, dim=1)
        size_feature = self.relu_1((self.conv_size_02(self.conv_size_01(x_0))))
        context_feature = self.relu_2((self.conv_context_02(self.conv_context_01(x_1))))
        return size_feature, context_feature  


class skip_blocks(nn.Module):

    def __init__(self, in_size, out_size, repeat_num=1):
        super(skip_blocks, self).__init__()
        self.blocks = nn.ModuleList()
        self.re_num = repeat_num
        mid_c = 128
        self.blocks.append(UNetConvBlock(in_size, mid_c, False, 0.2))
        for i in range(self.re_num - 2):
            self.blocks.append(UNetConvBlock(mid_c, mid_c, False, 0.2))
        self.blocks.append(UNetConvBlock(mid_c, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for m in self.blocks:
            x = m(x)
        return x + sc

def unfold_and_permute(tensor, kernel, stride=3, pad=-1):
    if pad < 0:
        pad = (kernel - 1) // 2
    tensor = F.pad(tensor, (pad, pad, pad, pad))
    tensor = tensor.unfold(2, kernel, stride)
    tensor = tensor.unfold(3, kernel, stride)
    N, C, H, W, _, _ = tensor.size()
    tensor = tensor.reshape(N, C, H, W, -1)
    tensor = tensor.permute(0, 2, 3, 1, 4)
    return tensor

def weight_permute_reshape(tensor, F, S2):
    N, C, H, W = tensor.size()
    tensor = tensor.permute(0, 2, 3, 1)
    tensor = tensor.reshape(N, H, W, F, S2)
    return tensor

# Filter_adaptive_convolution
def FAC(feat, filters, kernel_size, stride=3):
    N, C, H, W = feat.size()
    pad = (kernel_size - 1) // 2
    feat = unfold_and_permute(feat, kernel_size, stride, pad)
    weight = weight_permute_reshape(filters, C, kernel_size**2)

    output = feat * weight
    output = output.sum(-1)
    output = output.permute(0,3,1,2)
    return output


if __name__ == "__main__":
    pass
