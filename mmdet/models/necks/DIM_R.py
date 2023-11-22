import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from ..builder import NECKS


# 定义空间特征滤波器
class SpatialFeatureFilter(nn.Module):
    def forward(self, a_prime):
        # 计算全局平均池化
        global_avg_pool = torch.mean(a_prime, dim=(2, 3), keepdim=True)
        # 使用sigmoid函数对池化结果进行soften
        f_a_prime = 1 / (1 + torch.exp(-global_avg_pool))
        return f_a_prime

class AEM(nn.Module):
    def __init__(self):
        super(AEM, self).__init__()
        self.conv_p = nn.Conv2d(256*3, 256, kernel_size=1)
        self.conv_p1 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv_p2 = nn.Conv2d(256, 256, kernel_size=5, padding=4, groups=256)
        self.conv_p3 = nn.Conv2d(256, 256, kernel_size=5, padding=4, dilation=3, groups=256)
        self.conv_p4 = nn.Conv2d(256, 256, kernel_size=1)
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm2d(256)
        self.SSF = SpatialFeatureFilter()
        self.Deconv= nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0)
    def forward(self, x):
        p2, p3, p4 = x
        d4 = self.Deconv(p4)
        d3 = self.Deconv(p3)
        out_size = p2.shape[-2:]
        p3_1 = F.interpolate(p3, size=out_size, mode="nearest")
        p4_1 = F.interpolate(p4, size=out_size, mode="nearest")
        pc = torch.cat((p2, p3_1, p4_1), dim=1)
        pc = self.conv_p(pc)
        pc1 = self.conv_p1(pc)
        pc2 = self.conv_p2(pc1)
        pc3 = self.conv_p3(pc2)
        pc4 = self.conv_p4(pc3)
        pc5 = torch.mul(pc, pc4)
        pc5 = self.bn(pc5)
        ac = self.gelu(pc5)
        SFF_d4 = self.SSF(d4)
        SFF_d3 = self.SSF(d3)
        a4 = F.adaptive_max_pool2d(ac, output_size=p4.shape[-2:])
        a3 = F.adaptive_max_pool2d(ac, output_size=p3.shape[-2:]) * SFF_d4
        a2 = F.adaptive_max_pool2d(ac, output_size=p2.shape[-2:]) * SFF_d3
        r4 = a4 + p4
        r3 = a3 + p3
        r2 = a2 + p2
        R = [r2, r3, r4]
        return R



@NECKS.register_module()
class DIM_R(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
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
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
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
        super(DIM_R, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()


        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
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
        self.fpn_convs = nn.ModuleList()
        self.AEM = AEM()

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

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

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
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.conv_1x1_5 = nn.Conv2d(2048, 256, 1)
        self.conv_1x1_4 = nn.Conv2d(1024, 256, 1)
        self.conv_1x1_3 = nn.Conv2d(512, 256, 1)
        self.conv_1x1_2 = nn.Conv2d(256, 256, 1)
        self.conv_1x1_6 = nn.Conv2d(128, 256, 1)
        self.conv_1x1_7 = nn.Conv2d(64, 256, 1)
        self.conv_1x1_8 = nn.Conv2d(256, 256, kernel_size=4, stride=1)
        self.conv_1x1_9 = nn.Conv2d(256, 256, kernel_size=2, stride=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        # 经过SSF后的1x1卷积
        self.SSF_C5 = nn.Conv2d(512, 256, 1)
        self.SSF_C4 = nn.Conv2d(256, 256, 1)
    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        outs = self.AEM(outs)
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
        return tuple(outs)
