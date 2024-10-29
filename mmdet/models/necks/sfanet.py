# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS
from ..utils import _GlobalConvModule, _BoundaryRefineModule

# 定义网络模型
class FRM1(nn.Module):
    def __init__(self):
        super(FRM1, self).__init__()
        self.conv1 = nn.Conv2d(768, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 64, kernel_size=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 256, kernel_size=1)

    def forward(self, x1):
        # 对P1进行全局平均池化并插值得到Q1
        x_1 = self.conv1(x1)
        x_2 = self.conv2(x_1)
        x_2 = self.conv3(x_2)
        x_3 = self.conv4(x_2)
        x = self.conv5(x_3)

        return x


class FRM(nn.Module):
    def __init__(self, c):
        super(FRM, self).__init__()
        self.c = c
        # Define the required convolution layers in the constructor
        self.conv1 = nn.Conv2d(2 * c, c, kernel_size=1)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(c, c // 4, kernel_size=1)
        self.conv4 = nn.Conv2d(c // 4, c // 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(c // 4, c, kernel_size=1)
        self.conv6 = nn.Conv2d(c, c, kernel_size=3, padding=1)

    def forward(self, F1, P1):
        # Upsample P2 to have the same spatial size as P1
        F1_upsampled = F.interpolate(F1, size=P1.shape[-2:], mode='nearest')
        # Concatenate P2 and P1 along the channel dimension
        # F1 = P2_upsampled + P1
        F1 = torch.cat([F1_upsampled, P1], dim=1)

        # 1x1 convolution followed by 3x3 convolution to get F2'
        F1_1 = self.conv1(F1)
        F2_prime = self.conv2(F1_1)

        # 1x1 convolution followed by 3x3 convolution to get F2''
        F2_prime_1 = self.conv3(F2_prime)
        F2_double_prime = self.conv4(F2_prime_1)

        # 1x1 convolution followed by 3x3 convolution to get F2
        F2_double_prime_1 = self.conv5(F2_double_prime)
        F2 = self.conv6(F2_double_prime_1)

        return F2


class FeatureFusion1(nn.Module):
    def __init__(self, C):
        super(FeatureFusion1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.SELayer = SELayer(256)
        self.FRM1 = FRM1()

    def forward(self, P1, P2, Q):
        F1 = self.avg_pool(P2)

        F1 = self.FRM1(F1, P1, Q)

        # F2 = self.SELayer(F1)
        F2 = self.pool(F1)
        F2 = F2 + P2
        return F1, F2


class FeatureFusion(nn.Module):
    def __init__(self, C):
        super(FeatureFusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.SELayer = SELayer(256)
        self.FRM = FRM(256)

    def forward(self, P1, P2):

        F1 = self.avg_pool(P2)
        # 对池化后的特征图进行重复操作
        #b, c, _, _ = F1.size()
       # F1 = F1.repeat(1, 1, P2.size(2), P2.size(3))
        # F1 = F.interpolate(F1, size=P1.shape[-2:], mode='nearest')
        # F1 = F1 +P1
        # F1 = torch.cat([F1, P1], dim=1)
        F1 = self.FRM(F1, P1)

        # F2 = self.SELayer(F1)
        F2 = self.pool(F1)
        F2 = F2 + P2
        return F1, F2


class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Net(nn.Module):
    def __init__(self, c):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # P1 -> conv1 -> ReLU -> conv2 -> P2
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


@NECKS.register_module()
class GSNETFRM(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """


    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 cls_num = 10,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(GSNETFRM, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.cls_num = cls_num
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.feature_fusion = FeatureFusion(256)
        # self.feature_fusion1 = FeatureFusion1(256)
        self.Net = Net(256)
        self.conv0 = nn.ModuleList()
        self.conv1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.conv3 = nn.ModuleList()
        self.kernel_size = 11

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.lateral_convs1 = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.gcn_convs = nn.ModuleList()
        self.br_convs = nn.ModuleList()
        self.l1_convs = nn.ModuleList()
        self.se_blocks = SELayer(256)
        # self.FeatureInteraction = FeatureFusion(256)
        self.FRM1 = FRM1()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            conv0 = ConvModule(
                self.cls_num,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=False)
            conv2 = ConvModule(
                in_channels[i],
                self.cls_num,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                inplace=False)
            conv3 = _GlobalConvModule(
                in_channels[i],
                self.cls_num,
                (self.kernel_size, self.kernel_size)
            )
            #####################################
            conv4 = _BoundaryRefineModule(
                self.cls_num
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.lateral_convs1.append(conv0)
            self.l1_convs.append(conv2)
            self.gcn_convs.append(conv3)
            self.br_convs.append(conv4)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        l1_out_lays = [0]
        l1_out_lays[0] = self.l1_convs[0](inputs[0])
        # l1_out_lays = [
        #     l1_conv(inputs[i + self.start_level]) for i, l1_conv in enumerate(self.l1_convs)
        #     if i <= 0
        # ]
        # gcn___________________________________________________________________________________________________________
        gcn_out_lays = [0]
        gcn_out_lays[0] = self.gcn_convs[0](inputs[0])
        # gcn_out_lays = [
        #     gcn_conv(inputs[i + self.start_level]) for i, gcn_conv in enumerate(self.gcn_convs)
        #     if i <= 0
        # ]
        # gcn___________________________________________________________________________________________________________

        # br____________________________________________________________________________________________________________
        br_out_lays = [0]
        br_out_lays[0] = self.br_convs[0](gcn_out_lays[0])

        # br_out_lays = [
        #     br_conv(gcn_out_lays[i + self.start_level]) for i, br_conv in enumerate(self.br_convs)
        #     if i <= 0
        # ]
        # br____________________________________________________________________________________________________________
        # +
        for i in range(len(br_out_lays)):
            br_out_lays[i] = l1_out_lays[i] + br_out_lays[i]
        # 1x1___________________________________________________________________________________________________________
        laterals1 = self.lateral_convs1[0](br_out_lays[0])
        #


        # l1_out_lays = self.l1_convs[0](inputs[self.start_level])
        # # gcn___________________________________________________________________________________________________________
        # gcn_out_lays = self.gcn_convs[0](inputs[self.start_level])
        #
        #
        # # gcn___________________________________________________________________________________________________________
        #
        # # br____________________________________________________________________________________________________________
        # br_out_lays = self.br_convs[0](gcn_out_lays)
        #
        # # br____________________________________________________________________________________________________________
        # # +
        # br_out_lays = l1_out_lays + br_out_lays
        # # 1x1___________________________________________________________________________________________________________
        # laterals1 = [
        #     # inputs[i + self.start_level]:backbone输出的fmp
        #     # lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)
        #     self.lateral_convs1[0](br_out_lays)
        # ]

        laterals2 = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
            if i >= 1
        ]
        # laterals = laterals1 + laterals2

        # build top-down path
        used_backbone_levels1 = len(laterals2)
        for i in range(used_backbone_levels1 - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals2[i - 1] = laterals2[i - 1] + F.interpolate(
                    laterals2[i], **self.upsample_cfg)
            else:
                prev_shape = laterals2[i - 1].shape[2:]
                laterals2[i - 1] = laterals2[i - 1] + F.interpolate(
                    laterals2[i], size=prev_shape, **self.upsample_cfg)


        temp = F.interpolate(laterals2[0], scale_factor=2, mode='nearest')
        # print(f"[INFO] laterals[{i}].shape:{laterals[i].shape}")
        # print(f"[INFO] temp.shape:{temp.shape}")
        # laterals1_1 = laterals1[0]
        laterals1 = torch.cat([laterals1, temp, inputs[0]], dim=1)
        # print(f"[INFO] laterals[{i - 1}].shape:{laterals[i - 1].shape}")
        # laterals[i - 1] = self.make_five_conv_use(laterals[i - 1])
        laterals1_2 = self.FRM1(laterals1)
            # print(f"[INFO] laterals[{i - 1}].shape:{laterals[i - 1].shape}")
            # build outputs
            # part 1: from original levels
            # assert False
        laterals1_3 = [laterals1_2]
        laterals = laterals1_3 + laterals2
        # build outputs
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels1+1)
        ]

        used_backbone_levels = used_backbone_levels1+1
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        # outs[0:2] = self.feature_fusion(outs[0], outs[1])
        # outs[2:4] = self.feature_fusion(outs[2], outs[3])
        # outs[4] = self.se_blocks(outs[4])
        return tuple(outs)
