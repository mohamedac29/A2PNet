import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ConvBNReLU(nn.Module):
   

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class conv_block(nn.Module):
   

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=(1, 1), group=1, bn_act=False,
                 bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, group, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)
        self.use_bn_act = bn_act

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn_act:
            x = self.bn(x)
            x = self.act(x)
        return x


class DownsamplerBlock(nn.Module):
    

    def __init__(self, in_channels, out_channels, variant='M'):
        super().__init__()
        self.variant = variant
        if self.variant == 'L':
            # Large variant uses a dual-pooling method
            self.conv1 = conv_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            self.conv2 = conv_block(2 * out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
            self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False)
        else:
            # Other variants use a strided convolution
            self.conv = conv_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x):
        if self.variant == 'L':
            x = self.conv1(x)
            x1 = self.maxpool(x)
            x2 = self.avgpool(x)
            out = torch.cat([x1, x2], 1)
            return self.conv2(out)
        else:
            return self.conv(x)


class DWConvBlock(nn.Module):
    """Depthwise Separable Convolution Block"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, group=1,
                                bn_act=False)
        self.conv2 = conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                                group=out_channels, bn_act=False)
        self.bnorm = nn.BatchNorm2d(in_channels)
        self.cmixer = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=False)

    def forward(self, x):
        skip = x
        x = self.conv1(x)
        x = self.bnorm(x)
        x = self.conv2(x)
        x += skip
        return F.relu(x)


class FSM(nn.Module):
   

    def __init__(self, in_feats, out_feat):
        super(FSM, self).__init__()
        self.conv1_x = conv_block(in_feats, out_feat, kernel_size=1, stride=1, padding=0, bn_act=False)
        self.avg = nn.AvgPool2d(1)
        self.conv1 = conv_block(out_feat, out_feat, kernel_size=1, stride=1, padding=0, bn_act=False)
        self.bn = nn.BatchNorm2d(out_feat)
        self.sigmoid = nn.Sigmoid()
        self.dw_block = DWConvBlock(out_feat, out_feat)

    def forward(self, x):
        x = self.conv1_x(x)
        inp = self.avg(x)
        y = self.dw_block(inp)
        x = self.conv1(inp)
        x = self.bn(x)
        x = self.sigmoid(x)
        x = inp * x
        out = y + x
        return F.relu(out)


class SLKBlock(nn.Module):
    """Structured Large Kernel Block"""

    def __init__(self, in_channels, out_channels, k=3, drate=1):
        super().__init__()
        self.conv1 = conv_block(in_channels, in_channels, kernel_size=(k, k), stride=1, padding=drate, dilation=drate,
                                group=in_channels, bn_act=True)
        self.conv2 = conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, group=1,
                                bn_act=True)
        self.bnorm = nn.BatchNorm2d(in_channels)
        self.cmixer = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=False)

    def forward(self, x):
        skip = x
        x = self.conv1(x)
        x = self.cmixer(x)
        x = self.bnorm(x)
        x += skip
        return self.conv2(x)


class StemBlock(nn.Module):
    """Initial block to process the input image."""

    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv1 = conv_block(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bn_act=True)

    def forward(self, x):
        return self.conv1(x)


class ECA(nn.Module):
    """Efficient Channel Attention"""

    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class FDSSModule(nn.Module):
    

    def __init__(self, in_channels, out_channels, variant='M'):
        super().__init__()
        self.variant = variant
        self.conv3 = conv_block(in_channels, 1, kernel_size=3, stride=1, padding=1, bn_act=True)
        self.softmax = nn.Softmax(dim=1)
        self.conv2_1 = conv_block(in_channels, 1, kernel_size=1, stride=1, padding=0)
        # Fix: ECA channel is 1 as its input `att` has 1 channel.
        self.eca = ECA(channel=1)

    def forward(self, x):
        br1 = self.conv3(x)
        b, c, h, w = br1.size()  # c is 1
        br1 = self.softmax(br1)
        br1 = br1.view(b, c, -1)
        br2 = self.conv2_1(x)
        br2 = br2.view(b, c, -1).permute(0, 2, 1)
        att = torch.bmm(br1, br2)
        att = att.view(b, c, 1, 1)

        # 'tiny' variant skips the ECA module
        if self.variant != 'tiny':
            att = self.eca(att)
        return att


class FPAModule(nn.Module):

    def __init__(self, in_ch, out_ch, variant='M'):
        super().__init__()
        self.mid = ConvBNReLU(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

        # Use standard or depthwise convolutions based on variant
        if variant == 'L':
            self.down1 = ConvBNReLU(in_ch, in_ch, kernel_size=3, stride=2, padding=1)
            self.down2 = ConvBNReLU(in_ch, in_ch, kernel_size=3, stride=2, padding=1)
            self.down3 = ConvBNReLU(in_ch, in_ch, kernel_size=3, stride=2, padding=1)
        else:
            self.down1 = ConvBNReLU(in_ch, in_ch, kernel_size=3, stride=2, groups=in_ch, padding=1)
            self.down2 = ConvBNReLU(in_ch, in_ch, kernel_size=3, stride=2, groups=in_ch, padding=1)
            self.down3 = ConvBNReLU(in_ch, in_ch, kernel_size=3, stride=2, groups=in_ch, padding=1)

        self.tran1 = FDSSModule(in_ch, out_ch, variant=variant)
        self.tran2 = FDSSModule(in_ch, out_ch, variant=variant)
        self.tran3 = FDSSModule(in_ch, out_ch, variant=variant)

        self.conv2 = ConvBNReLU(1, out_ch, kernel_size=1, stride=1, padding=0)
        self.out = ConvBNReLU(out_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        mid = self.mid(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        tran1 = self.tran1(x1)
        tran2 = self.tran2(x2)
        tran3 = self.tran3(x3)

        up1 = F.interpolate(tran3, size=tran2.size()[2:], mode='bilinear', align_corners=True)
        up1 = up1 + tran2
        up2 = F.interpolate(up1, size=tran1.size()[2:], mode='bilinear', align_corners=True)
        up2 = up2 + tran1
        up2 = F.interpolate(up2, size=mid.size()[2:], mode='bilinear', align_corners=True)

        up2 = self.conv2(up2)
        out = up2 + mid
        out = self.out(out)
        return out


class EfficientDecoderBlock(nn.Module):

    def __init__(self, l_feats, h_feats, out_channels):
        super(EfficientDecoderBlock, self).__init__()
        self.convx = conv_block(l_feats, out_channels, 1, 1, padding=0, bn_act=True)
        self.convy = nn.Sequential(
            nn.AvgPool2d(1),
            conv_block(h_feats, out_channels, 1, 1, padding=0, bn_act=True),
            nn.Sigmoid()
        )
        self.fconv = conv_block(2 * out_channels, out_channels, 3, 1, padding=1, bn_act=True)

    def forward(self, x, y):
        x = self.convx(x)
        y = self.convy(y)
        y = F.interpolate(y, size=x.size()[2:], mode='bilinear', align_corners=False)
        fuse = torch.cat([x, y], 1)
        fuse = self.fconv(fuse)
        return fuse


class A2PNet(nn.Module):
    """
    Unified A2PNet model with selectable variants.

    Args:
        variant (str): The model variant to create.
                       Options: 'tiny', 'S', 'M', 'L'.
        img_ch (int): Number of input image channels.
        output_ch (int): Number of output channels for segmentation.
    """

    def __init__(self, variant='M', img_ch=1, output_ch=1):
        super().__init__()

        # Configurations for each model variant
        model_configs = {
            'tiny': {'dims': [8, 16, 24, 48, 64]},
            'S': {'dims': [16, 48, 64, 96, 160]},
            'M': {'dims': [24, 48, 96, 160, 256]},
            'L': {'dims': [32, 128, 160, 192, 384]},
        }

        if variant not in model_configs:
            raise ValueError(f"Variant '{variant}' not recognized. Available variants: {list(model_configs.keys())}")

        config = model_configs[variant]
        dims = config['dims']

        # Block and kernel sizes are consistent across variants
        block_1, block_2, block_3, block_4, block_5 = 2, 6, 6, 2, 3
        kernels = [3, 3, 3, 3, 3]

        # --- Encoder ---
        self.stem = StemBlock(img_ch, dims[0], stride=2)
        self.stage1 = self._make_layer(SLKBlock, dims[0], dims[0], block_1, kernels[0])
        self.down1 = DownsamplerBlock(dims[0], dims[1], variant=variant)
        self.stage2 = self._make_layer(SLKBlock, dims[1], dims[1], block_2, kernels[1])
        self.down2 = DownsamplerBlock(dims[1], dims[2], variant=variant)
        self.stage3 = self._make_layer(SLKBlock, dims[2], dims[2], block_3, kernels[2])
        self.down3 = DownsamplerBlock(dims[2], dims[3], variant=variant)
        self.stage4 = self._make_layer(SLKBlock, dims[3], dims[3], block_4, kernels[3], dilation_rates=[2, 4, 8, 16])
        self.down4 = conv_block(dims[3], dims[4], 3, 1, padding=1, bn_act=True)
        self.stage5 = self._make_layer(SLKBlock, dims[4], dims[4], block_5, kernels[4], dilation_rates=[3, 7, 9, 13])

        # --- Context Module ---
        self.context = nn.Sequential(
            conv_block(dims[4], 64, 3, 1, padding=1, bn_act=True),
            FPAModule(64, 64, variant=variant)
        )

        # --- Decoder Bridges ---
        self.mid4 = FSM(dims[3], 64)
        self.mid3 = FSM(dims[2], 64)
        self.mid2 = FSM(dims[1], 64)
        self.mid1 = FSM(dims[0], 64)

        # --- Decoder ---
        self.dec4 = EfficientDecoderBlock(64, 64, 64)
        self.dec3 = EfficientDecoderBlock(64, 64, 64)
        self.dec2 = EfficientDecoderBlock(64, 64, 64)
        self.dec1 = EfficientDecoderBlock(64, 64, 64)

        # --- Segmentation Head ---
        self.final_seg = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, output_ch, kernel_size=1)
        )
        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, block, in_channels, out_channels, blocks, k, dilation_rates=None):
        """Helper function to create a sequence of blocks."""
        layers = []
        for i in range(blocks):
            drate = dilation_rates[i % len(dilation_rates)] if dilation_rates else 1
            layers.append(block(in_channels, out_channels, k=k, drate=drate))
        return nn.Sequential(*layers)

    def forward(self, x):
        B, C, H, W = x.size()

        # Encoder path
        stem = self.stem(x)
        s1 = self.stage1(stem)
        s2 = self.stage2(self.down1(s1))
        s3 = self.stage3(self.down2(s2))
        s4 = self.stage4(self.down3(s3))
        s5 = self.stage5(self.down4(s4))

        context = self.context(s5)

        # Decoder path with skip connections
        mid4 = self.mid4(s4)
        mid3 = self.mid3(s3)
        mid2 = self.mid2(s2)
        mid1 = self.mid1(s1)

        dec4 = self.dec4(mid4, context)
        dec3 = self.dec3(mid3, dec4)
        dec2 = self.dec2(mid2, dec3)
        dec1 = self.dec1(mid1, dec2)

        # Final segmentation output
        seg_out = self.final_seg(dec1)
        seg_out = F.interpolate(seg_out, size=(H, W), mode="bilinear", align_corners=True)
        return self.softmax(seg_out)


if __name__ == "__main__":
    # Example of how to use the unified A2PNet model

    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Input tensor
    input_tensor = torch.rand(2, 1, 512, 512).to(device)

    print("\n--- A2PNet Tiny ---")
    model_tiny = A2PNet(variant='tiny', img_ch=1, output_ch=2).to(device)
    model_tiny.eval()
    summary(model_tiny, (1, 512, 512))

    print("\n--- A2PNet Small (S) ---")
    model_s = A2PNet(variant='S', img_ch=1, output_ch=2).to(device)
    model_s.eval()
    summary(model_s, (1, 512, 512))

    print("\n--- A2PNet Medium (M) ---")
    model_m = A2PNet(variant='M', img_ch=1, output_ch=2).to(device)
    model_m.eval()
    summary(model_m, (1, 512, 512))
    #
    print("\n--- A2PNet Large (L) ---")
    model_l = A2PNet(variant='L', img_ch=1, output_ch=2).to(device)
    model_l.eval()
    summary(model_l, (1, 512, 512))

    # # Test forward pass
    # with torch.no_grad():
    #     output = model_l(input_tensor)
    # print(f"\nOutput shape for Large model: {output.shape}")
