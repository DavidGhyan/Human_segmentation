import torch
import torch.nn as nn
from mobile import CBG, MobileNetv, MobileNetV1Block,MobileNetV2Block


# class CBG(nn.Module):
#     def __init__(self, in_nc, out_nc, kernel_size=3, stride=1):
#         super(CBG, self).__init__()
#         self.padding = kernel_size - 1  # Reflective padding to preserve dimensions after convolution
#         self.conv = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=0)  # Convolution layer
#         self.bn = nn.BatchNorm2d(out_nc)  # Batch normalization
#         self.gelu = nn.GELU()  # GELU activation function
#
#     def forward(self, x):
#         x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')  # Apply reflect padding
#         out = self.conv(x)  # Convolution operation
#         out = self.bn(out)  # Normalize activations
#         out = self.gelu(out)  # Apply GELU activation
#         return out


class ConvBlock(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            CBG(in_nc, out_nc, kernel_size=kernel_size, stride=stride),
            CBG(out_nc, out_nc, kernel_size=kernel_size, stride=stride)
        )

    def forward(self, inputs):
        return self.conv_block(inputs)


class Downsample_Block(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3, stride=1):
        super(Downsample_Block, self).__init__()
        self.conv_block = ConvBlock(in_nc, out_nc, kernel_size=kernel_size, stride=stride)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling for downsampling

    def forward(self, inputs):
        x = self.conv_block(inputs)
        p = self.maxpool(x)  # Apply max pooling for spatial downsampling
        return x, p


class Upsample_Block(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3, stride=1, skip_connect=True):
        super(Upsample_Block, self).__init__()
        self.skip_connect = skip_connect  # Set skip_connect flag
        self.up = nn.ConvTranspose2d(in_nc, out_nc, kernel_size=kernel_size, stride=2)  # Transposed conv for upsampling
        if self.skip_connect:
            self.conv_block = ConvBlock(2 * out_nc, out_nc, kernel_size=kernel_size,
                                        stride=stride)  # For skip connection
        else:
            self.conv_block = ConvBlock(out_nc, out_nc, kernel_size=kernel_size, stride=stride)

    def forward(self, inputs, skip=None):
        x = self.up(inputs)  # Upsample the input tensor
        if self.skip_connect:
            if skip is None:
                raise ValueError("skip connection is required when skip_connect=True")
            x = torch.cat([x, skip], axis=1)  # Concatenate the skip connection
        return self.conv_block(x)


class UNet(nn.Module):
    def __init__(self, n_channels, num_classes):
        super(UNet, self).__init__()

        # Downsampling (encoder) blocks
        self.down1 = Downsample_Block(n_channels, 64)
        self.down2 = Downsample_Block(64, 128)
        self.down3 = Downsample_Block(128, 256)
        self.down4 = Downsample_Block(256, 512)

        # Bottleneck block
        self.b = ConvBlock(512, 1024)

        # Upsampling (decoder) blocks
        self.up1 = Upsample_Block(1024, 512, skip_connect=False)  # Here skip_connect=False
        self.up2 = Upsample_Block(512, 256)
        self.up3 = Upsample_Block(256, 128)
        self.up4 = Upsample_Block(128, 64)

        # Final output layer
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, inputs):
        s1, p1 = self.down1(inputs)
        s2, p2 = self.down2(p1)
        s3, p3 = self.down3(p2)
        s4, p4 = self.down4(p3)

        b = self.b(p4)

        # For the first upsampling, skip_connect=False, no skip connections
        u1 = self.up1(b)  # Without skip connections
        u2 = self.up2(u1, s3)
        u3 = self.up3(u2, s2)
        u4 = self.up4(u3, s1)

        outputs = self.out(u4)
        return outputs


class UNet_MobileNetv1(nn.Module):
    def __init__(self, n_channels, num_classes):
        super(UNet_MobileNetv1, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes

        # ---------------------------------------------------#
        #   128x128x128; 64x64x256; 32x32x512, 16x16x1024
        # ---------------------------------------------------#
        # Backbone MobileNetv1
        self.backbone = MobileNetv(n_channels,MobileNetV1Block)

        self.up1 = Upsample_Block(1024, 512, skip_connect=False)  # Here skip_connect=False
        self.up2 = Upsample_Block(512, 256)
        self.up3 = Upsample_Block(256, 128)
        self.up4 = Upsample_Block(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Extract features using MobileNetv1
        s1, s2, s3, s4 = self.backbone(x)

        # Upsampling with skip connections
        u1 = self.up1(s4)  # Here no skip connections
        u2 = self.up2(u1, s3)
        u3 = self.up3(u2, s2)
        u4 = self.up4(u3, s1)

        # Final output layer
        outputs = self.out(u4)
        return outputs
