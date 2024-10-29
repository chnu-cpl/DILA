import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
from mmcv.ops import box_iou_rotated, DeformConvG, DeformConv2d, ModulatedDeformConv2d, ModulatedDeformConvG
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from mmdet.core import anchor, images_to_levels, multi_apply, unmap, reduce_mean

from ..builder import HEADS
from .anchor_head import AnchorHead


@HEADS.register_module()
class DCFLRetinaHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 dcn_assign=False,
                 dilation_rate=2,  #
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn_assign = dcn_assign #
        self.dilation_rate = dilation_rate  #
        super(DCFLRetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        if self.dcn_assign == False:
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                self.cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
                self.reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
        else:
            for i in range(self.stacked_convs - 2):  #
                chn = self.in_channels if i == 0 else self.feat_channels
                self.cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
                self.reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            for i in range(1):  #
                self.cls_convs.append(
                    DeformConv2d(
                        self.feat_channels,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        groups=1,
                        bias=False))
                self.reg_convs.append(
                    DeformConv2d(
                        self.feat_channels,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        groups=1,
                        bias=False))
            for i in range(1):  #
                self.cls_convs.append(
                    ModulatedDeformConv2d(
                        self.feat_channels,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        groups=1,
                        bias=False))
                self.reg_convs.append(
                    ModulatedDeformConvG(
                    self.feat_channels,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        groups=1,
                        bias=False))

        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        if self.dcn_assign == False:  #
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
            offsets_reg = None
        else:
            for reg_conv in self.reg_convs[:-2]:
                reg_feat = reg_conv(reg_feat)

            init_t = torch.Tensor(reg_feat.size(0), 1, reg_feat.size(-2), reg_feat.size(-1))
            item = torch.ones_like(init_t, device=reg_feat.device) * (self.dilation_rate - 1)
            zeros = torch.zeros_like(item, device=reg_feat.device)
            sampling_loc = torch.cat((-item, -item, -item, zeros, -item, item, zeros, -item, zeros, zeros, zeros, item,
                                      item, -item, item, zeros, item, item), dim=1)

            reg_feat = self.reg_convs[self.stacked_convs - 2](reg_feat, sampling_loc)
            reg_feat, offsets_reg, mask_reg = self.reg_convs[self.stacked_convs - 1](reg_feat)

            for cls_conv in self.cls_convs[:-2]:
                cls_feat = cls_conv(cls_feat)
            cls_feat = self.cls_convs[self.stacked_convs - 2](cls_feat, sampling_loc)
            cls_feat = self.cls_convs[self.stacked_convs - 1](cls_feat, offsets_reg, mask_reg)

        # offset batch_size * 18 * feature_size * feature_size (128,64,32,16,8), offset [y0, x0, y1, x1, y2, x2, ..., y8, x8]
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred
        # return cls_score, bbox_pred, offsets_reg

    # def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
    #                 bbox_targets, bbox_weights, num_total_samples):
    #     """Compute loss of a single scale level.
    # 
    #     Args:
    #         cls_score (Tensor): Box scores for each scale level
    #             Has shape (N, num_anchors * num_classes, H, W).
    #         bbox_pred (Tensor): Box energies / deltas for each scale
    #             level with shape (N, num_anchors * 4, H, W).
    #         anchors (Tensor): Box reference for each scale level with shape
    #             (N, num_total_anchors, 4).
    #         labels (Tensor): Labels of each anchors with shape
    #             (N, num_total_anchors).
    #         label_weights (Tensor): Label weights of each anchor with shape
    #             (N, num_total_anchors)
    #         bbox_targets (Tensor): BBox regression targets of each anchor wight
    #             shape (N, num_total_anchors, 4).
    #         bbox_weights (Tensor): BBox regression loss weights of each anchor
    #             with shape (N, num_total_anchors, 4).
    #         num_total_samples (int): If sampling, num total samples equal to
    #             the number of total anchors; Otherwise, it is the number of
    #             positive anchors.
    # 
    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components.
    #     """
    #     # classification loss
    #     labels = labels.reshape(-1)
    #     label_weights = label_weights.reshape(-1)
    #     cls_score = cls_score.permute(0, 2, 3,
    #                                   1).reshape(-1, self.cls_out_channels)
    #     loss_cls = self.loss_cls(
    #         cls_score, labels, label_weights, avg_factor=num_total_samples)
    #     # regression loss
    #     bbox_targets = bbox_targets.reshape(-1, 4)
    #     bbox_weights = bbox_weights.reshape(-1, 4)
    #     bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
    #     if self.reg_decoded_bbox:
    #         # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
    #         # is applied directly on the decoded bounding boxes, it
    #         # decodes the already encoded coordinates to absolute format.
    #         anchors = anchors.reshape(-1, 4)
    #         bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
    #     loss_bbox = self.loss_bbox(
    #         bbox_pred,
    #         bbox_targets,
    #         bbox_weights,
    #         avg_factor=num_total_samples)
    #     return loss_cls, loss_bbox
    # 
    # @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    # def loss(self,
    #          cls_scores,
    #          bbox_preds,
    #          offsets,
    #          gt_bboxes,
    #          gt_labels,
    #          img_metas,
    #          gt_bboxes_ignore=None):
    #     """Compute losses of the head.
    # 
    #     Args:
    #         cls_scores (list[Tensor]): Box scores for each scale level
    #             Has shape (N, num_anchors * num_classes, H, W)
    #         bbox_preds (list[Tensor]): Box energies / deltas for each scale
    #             level with shape (N, num_anchors * 4, H, W)
    #         gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
    #             shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
    #         gt_labels (list[Tensor]): class indices corresponding to each box
    #         img_metas (list[dict]): Meta information of each image, e.g.,
    #             image size, scaling factor, etc.
    #         gt_bboxes_ignore (None | list[Tensor]): specify which bounding
    #             boxes can be ignored when computing the loss. Default: None
    # 
    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components.
    #     """
    #     featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    #     assert len(featmap_sizes) == self.anchor_generator.num_levels
    # 
    #     device = cls_scores[0].device
    # 
    #     anchor_list, valid_flag_list = self.get_anchors(
    #         featmap_sizes, img_metas, device=device)
    #     label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
    # 
    #     cls_reg_targets = self.get_targets(
    #         cls_scores,#
    #         bbox_preds,#
    #         anchor_list,
    #         valid_flag_list,
    #         offsets,#
    #         gt_bboxes,
    #         img_metas,
    #         gt_bboxes_ignore_list=gt_bboxes_ignore,
    #         gt_labels_list=gt_labels,
    #         label_channels=label_channels)
    #     if cls_reg_targets is None:
    #         return None
    #     (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
    #      num_total_pos, num_total_neg) = cls_reg_targets
    #     num_total_samples = (
    #         num_total_pos + num_total_neg if self.sampling else num_total_pos)
    # 
    #     # anchor number of multi levels
    #     num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    #     # concat all level anchors and flags to a single tensor
    #     concat_anchor_list = []
    #     for i in range(len(anchor_list)):
    #         concat_anchor_list.append(torch.cat(anchor_list[i]))
    #     all_anchor_list = images_to_levels(concat_anchor_list,
    #                                        num_level_anchors)
    # 
    #     losses_cls, losses_bbox = multi_apply(
    #         self.loss_single,
    #         cls_scores,
    #         bbox_preds,
    #         all_anchor_list,
    #         labels_list,
    #         label_weights_list,
    #         bbox_targets_list,
    #         bbox_weights_list,
    #         num_total_samples=num_total_samples)
    #     return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
    # 
    # def _get_targets_single(self,
    #                         flat_cls_scores,
    #                         flat_bbox_preds,
    #                         flat_anchors,
    #                         valid_flags,
    #                         offsets,
    #                         offsets_ori,
    #                         gt_bboxes,
    #                         gt_bboxes_ignore,
    #                         gt_labels,
    #                         img_meta,
    #                         label_channels=1,
    #                         unmap_outputs=False):
    #     """Compute regression and classification targets for anchors in a
    #                 single image.
    # 
    #                 Args:
    #                     flat_anchors (Tensor): Multi-level anchors of the image, which are
    #                         concatenated into a single tensor of shape (num_anchors ,4)
    #                     valid_flags (Tensor): Multi level valid flags of the image,
    #                         which are concatenated into a single tensor of
    #                             shape (num_anchors,).
    #                     gt_bboxes (Tensor): Ground truth bboxes of the image,
    #                         shape (num_gts, 4).
    #                     gt_bboxes_ignore (Tensor): Ground truth bboxes to be
    #                         ignored, shape (num_ignored_gts, 4).
    #                     img_meta (dict): Meta info of the image.
    #                     gt_labels (Tensor): Ground truth labels of each box,
    #                         shape (num_gts,).
    #                     label_channels (int): Channel of label.
    #                     unmap_outputs (bool): Whether to map outputs back to the original
    #                         set of anchors.
    # 
    #                 Returns:
    #                     tuple:
    #                         labels_list (list[Tensor]): Labels of each level
    #                         label_weights_list (list[Tensor]): Label weights of each level
    #                         bbox_targets_list (list[Tensor]): BBox targets of each level
    #                         bbox_weights_list (list[Tensor]): BBox weights of each level
    #                         num_total_pos (int): Number of positive samples in all images
    #                         num_total_neg (int): Number of negative samples in all images
    #                 """
    #     inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
    #                                        img_meta['img_shape'][:2],
    #                                        self.train_cfg.allowed_border)
    #     if not inside_flags.any():
    #         return (None,) * 7
    #     # assign gt and sample anchors
    #     anchors = flat_anchors #
    #     # anchors = flat_anchors[inside_flags, :]
    #     # print(offsets.size())
    #     detached_offsets = offsets.detach()  #
    #     dy_ori = torch.zeros(1, detached_offsets.size(1)).cuda()
    #     dx_ori = torch.zeros(1, detached_offsets.size(1)).cuda()
    #     # print(offsets.size())
    #     # print(dx.size())
    # 
    #     for i in range(9):
    #         dy_ori += detached_offsets[2 * i] / 9
    #         dx_ori += detached_offsets[2 * i + 1] / 9
    #     # print(dx_ori.size())
    #     num1 = flat_anchors.size(0)
    #     dx = torch.zeros(1, num1).cuda()
    #     dy = torch.zeros(1, num1).cuda()
    #     num = dx_ori.size(1)
    #     dx[:, :num] = dx_ori
    #     dy[:, :num] = dy_ori
    #     # print(flat_anchors.size())
    #     # result = flat_anchors.clone()
    #     # flat_anchors[..., 0][:30685] += dx[0,...]
    #     # flat_anchors[..., 1][:30685] += dy[0,...]
    # 
    #     flat_anchors[..., 0] = flat_anchors[..., 0] + dx
    #     flat_anchors[..., 1] = flat_anchors[..., 1] + dy
    # 
    #     deformable_anchors = flat_anchors
    # 
    #     assign_result = self.assigner.assign(
    #         deformable_anchors, gt_bboxes, gt_bboxes_ignore,
    #         None if self.sampling else gt_labels)
    #     sampling_result = self.sampler.sample(assign_result, anchors,
    #                                           gt_bboxes)
    # 
    #     num_valid_anchors = anchors.shape[0]
    #     bbox_targets = torch.zeros_like(anchors)
    #     bbox_weights = torch.zeros_like(anchors)
    #     # print(anchors.size())
    #     labels = anchors.new_full((num_valid_anchors,),
    #                               self.num_classes,
    #                               dtype=torch.long)
    #     # print(labels.size())
    #     label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
    # 
    #     pos_inds = sampling_result.pos_inds
    #     neg_inds = sampling_result.neg_inds
    #     if len(pos_inds) > 0:
    #         if not self.reg_decoded_bbox:
    #             pos_bbox_targets = self.bbox_coder.encode(
    #                 sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
    #         else:
    #             pos_bbox_targets = sampling_result.pos_gt_bboxes
    #         bbox_targets[pos_inds, :] = pos_bbox_targets
    #         bbox_weights[pos_inds, :] = 1.0
    #         if gt_labels is None:
    #             # Only rpn gives gt_labels as None
    #             # Foreground is the first class since v2.5.0
    #             labels[pos_inds] = 0
    #         else:
    #             labels[pos_inds] = gt_labels[
    #                 sampling_result.pos_assigned_gt_inds]
    #         if self.train_cfg.pos_weight <= 0:
    #             label_weights[pos_inds] = 1.0
    #         else:
    #             label_weights[pos_inds] = self.train_cfg.pos_weight
    #     if len(neg_inds) > 0:
    #         label_weights[neg_inds] = 1.0
    # 
    #     # map up to original set of anchors
    #     if unmap_outputs:
    #         # num_total_anchors = flat_anchors.size(0)
    #         num_total_anchors = labels.size(0)
    #         # print(flat_anchors.size(), labels.size(), inside_flags.size())
    #         labels = unmap(
    #             labels, num_total_anchors, inside_flags,
    #             fill=self.num_classes)  # fill bg label
    #         label_weights = unmap(label_weights, num_total_anchors,
    #                               inside_flags)
    #         bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
    #         bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
    # 
    #     return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
    #             neg_inds, sampling_result)
    # 
    # def get_targets(self,
    #                 cls_scores_list,
    #                 bbox_pred_list,
    #                 anchor_list,
    #                 valid_flag_list,
    #                 offsets,
    #                 gt_bboxes_list,
    #                 img_metas,
    #                 gt_bboxes_ignore_list=None,
    #                 gt_labels_list=None,
    #                 label_channels=1,
    #                 unmap_outputs=False,
    #                 return_sampling_results=False):
    #     """Compute regression and classification targets for anchors in
    #     multiple images.
    # 
    #     Args:
    #         anchor_list (list[list[Tensor]]): Multi level anchors of each
    #             image. The outer list indicates images, and the inner list
    #             corresponds to feature levels of the image. Each element of
    #             the inner list is a tensor of shape (num_anchors, 5).
    #         valid_flag_list (list[list[Tensor]]): Multi level valid flags of
    #             each image. The outer list indicates images, and the inner list
    #             corresponds to feature levels of the image. Each element of
    #             the inner list is a tensor of shape (num_anchors, )
    #         offsets (list[list[Tensor]]): Offsets of DCN.
    #         gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
    #         img_metas (list[dict]): Meta info of each image.
    #         gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
    #             ignored.
    #         gt_labels_list (list[Tensor]): Ground truth labels of each box.
    #         label_channels (int): Channel of label.
    #         unmap_outputs (bool): Whether to map outputs back to the original
    #             set of anchors.
    # 
    #     Returns:
    #         tuple: Usually returns a tuple containing learning targets.
    # 
    #             - labels_list (list[Tensor]): Labels of each level.
    #             - label_weights_list (list[Tensor]): Label weights of each \
    #                 level.
    #             - bbox_targets_list (list[Tensor]): BBox targets of each level.
    #             - bbox_weights_list (list[Tensor]): BBox weights of each level.
    #             - num_total_pos (int): Number of positive samples in all \
    #                 images.
    #             - num_total_neg (int): Number of negative samples in all \
    #                 images.
    # 
    #         additional_returns: This function enables user-defined returns from
    #             `self._get_targets_single`. These returns are currently refined
    #             to properties at each feature map (i.e. having HxW dimension).
    #             The results will be concatenated after the end
    #     """
    #     num_imgs = len(img_metas)
    #     assert len(anchor_list) == len(valid_flag_list) == num_imgs
    # 
    #     # anchor number of multi levels, [128^2, 64^2, 32^2, 16^2, 8^2]
    #     num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # 
    #     # concat all level anchors to a single tensor
    # 
    #     concat_anchor_list = []
    #     concat_valid_flag_list = []
    #     for i in range(num_imgs):
    #         assert len(anchor_list[i]) == len(valid_flag_list[i])
    #         concat_anchor_list.append(
    #             torch.cat(anchor_list[i]))  # a list whose len is batch size, each element is a tensor
    #         concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))
    # 
    # 
    #     # 创建用于存储所有级别偏移量数据的列表
    #     concat_offsets = []
    #     concat_offsets_ori = []
    #     lvl_offsets = []
    #     lvl_offsets_ori = []
    # 
    #     # 计算用于缩放偏移量的因子，这个因子通常与图像的缩放有关
    #     factor = img_metas[0]['img_shape'][0] / 256
    # 
    #     # 创建用于存储所有级别分类分数和边界框预测的列表
    #     lvl_scores = []
    #     lvl_bboxes = []
    #     concat_cls_scores_list = []
    #     concat_bbox_pred_list = []
    # 
    #     # 遍历每个级别的分类分数和边界框预测
    #     for i in range(len(cls_scores_list)):
    #         reshaped_scores = cls_scores_list[i].detach().reshape(num_imgs, self.num_classes, -1)  # 重塑分类分数的形状
    #         reshaped_bboxes = bbox_pred_list[i].detach().reshape(num_imgs, 4, -1)  # 重塑边界框预测的形状
    #         lvl_scores.append(reshaped_scores)  # 将重塑后的分类分数添加到列表中
    #         lvl_bboxes.append(reshaped_bboxes)  # 将重塑后的边界框预测添加到列表中
    # 
    #     # 拼接所有级别的分类分数和边界框预测
    #     cat_lvl_scores = torch.cat(lvl_scores, dim=-1)  # 在最后一个维度上拼接分类分数
    #     cat_lvl_bboxes = torch.cat(lvl_bboxes, dim=-1)  # 在最后一个维度上拼接边界框预测
    # 
    #     # 遍历每个图像
    #     for j in range(num_imgs):
    #         concat_cls_scores_list.append(cat_lvl_scores[j, ...])  # 将拼接后的分类分数添加到列表中
    #         concat_bbox_pred_list.append(cat_lvl_bboxes[j, ...])  # 将拼接后的边界框预测添加到列表中
    # 
    #     # 遍历每个级别的偏移量数据，并缩放每个级别的偏移量
    #     for k in range(len(offsets)):
    #         # for i, offset in enumerate(offsets):
    #         #     if offsets is None:
    #         #         raise ValueError(f"offsets[{i}] is None!")
    #         reshaped_offsets_ori = offsets[k].reshape(num_imgs, 18, -1)  # 重塑原始偏移量的形状
    #         reshaped_offsets = reshaped_offsets_ori * factor  # 将偏移量乘以缩放因子
    #         lvl_offsets_ori.append(reshaped_offsets_ori)  # 将原始偏移量添加到列表中
    #         lvl_offsets.append(reshaped_offsets)  # 将缩放后的偏移量添加到列表中
    #         factor = factor * 2  # 更新缩放因子，通常是每个级别的偏移量缩放不同
    # 
    #     # 拼接所有级别的偏移量
    #     cat_lvl_offsets = torch.cat(lvl_offsets, dim=2)
    # 
    #     # 遍历每个图像，并将拼接后的偏移量添加到列表中
    #     for j in range(num_imgs):
    #         concat_offsets.append(cat_lvl_offsets[j, ...])
    # 
    #     # 拼接所有级别的原始偏移量
    #     cat_lvl_offsets_ori = torch.cat(lvl_offsets_ori, dim=2)
    # 
    #     # 遍历每个图像，并将拼接后的原始偏移量添加到列表中
    #     for j in range(num_imgs):
    #         concat_offsets_ori.append(cat_lvl_offsets_ori[j, ...])
    # 
    #     # compute targets for each image
    #     if gt_bboxes_ignore_list is None:
    #         gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    #     if gt_labels_list is None:
    #         gt_labels_list = [None for _ in range(num_imgs)]
    #     results = multi_apply(
    #         self._get_targets_single,
    #         concat_cls_scores_list,
    #         concat_bbox_pred_list,
    #         concat_anchor_list,
    #         concat_valid_flag_list,
    #         concat_offsets,
    #         concat_offsets_ori,
    #         gt_bboxes_list,
    #         gt_bboxes_ignore_list,
    #         gt_labels_list,
    #         img_metas,
    #         label_channels=label_channels,
    #         unmap_outputs=unmap_outputs)
    #     (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
    #      pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
    #     rest_results = list(results[7:])  # user-added return values
    #     # no valid anchors
    #     if any([labels is None for labels in all_labels]):
    #         return None
    #     # sampled anchors of all images
    #     num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    #     num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    #     # split targets to a list w.r.t. multiple levels
    #     labels_list = images_to_levels(all_labels, num_level_anchors)
    #     label_weights_list = images_to_levels(all_label_weights,
    #                                           num_level_anchors)
    #     bbox_targets_list = images_to_levels(all_bbox_targets,
    #                                          num_level_anchors)
    #     bbox_weights_list = images_to_levels(all_bbox_weights,
    #                                          num_level_anchors)
    #     res = (labels_list, label_weights_list, bbox_targets_list,
    #            bbox_weights_list, num_total_pos, num_total_neg)
    #     if return_sampling_results:
    #         res = res + (sampling_results_list,)
    #     for i, r in enumerate(rest_results):  # user-added return values
    #         rest_results[i] = images_to_levels(r, num_level_anchors)
    # 
    #     return res + tuple(rest_results)
    # 
    # @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    # def get_bboxes(self,
    #                cls_scores,
    #                bbox_preds,
    #                offsets,
    #                img_metas,
    #                cfg=None,
    #                rescale=False,
    #                with_nms=True):
    #     """Transform network output for a batch into bbox predictions.
    # 
    #     Args:
    #         cls_scores (list[Tensor]): Box scores for each level in the
    #             feature pyramid, has shape
    #             (N, num_anchors * num_classes, H, W).
    #         bbox_preds (list[Tensor]): Box energies / deltas for each
    #             level in the feature pyramid, has shape
    #             (N, num_anchors * 4, H, W).
    #         img_metas (list[dict]): Meta information of each image, e.g.,
    #             image size, scaling factor, etc.
    #         cfg (mmcv.Config | None): Test / postprocessing configuration,
    #             if None, test_cfg would be used
    #         rescale (bool): If True, return boxes in original image space.
    #             Default: False.
    #         with_nms (bool): If True, do nms before return boxes.
    #             Default: True.
    # 
    #     Returns:
    #         list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
    #             The first item is an (n, 5) tensor, where 5 represent
    #             (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
    #             The shape of the second tensor in the tuple is (n,), and
    #             each element represents the class label of the corresponding
    #             box.
    # 
    #     Example:
    #         >>> import mmcv
    #         >>> self = AnchorHead(
    #         >>>     num_classes=9,
    #         >>>     in_channels=1,
    #         >>>     anchor_generator=dict(
    #         >>>         type='AnchorGenerator',
    #         >>>         scales=[8],
    #         >>>         ratios=[0.5, 1.0, 2.0],
    #         >>>         strides=[4,]))
    #         >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
    #         >>> cfg = mmcv.Config(dict(
    #         >>>     score_thr=0.00,
    #         >>>     nms=dict(type='nms', iou_thr=1.0),
    #         >>>     max_per_img=10))
    #         >>> feat = torch.rand(1, 1, 3, 3)
    #         >>> cls_score, bbox_pred = self.forward_single(feat)
    #         >>> # note the input lists are over different levels, not images
    #         >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
    #         >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
    #         >>>                               img_metas, cfg)
    #         >>> det_bboxes, det_labels = result_list[0]
    #         >>> assert len(result_list) == 1
    #         >>> assert det_bboxes.shape[1] == 5
    #         >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
    #     """
    #     assert len(cls_scores) == len(bbox_preds)
    #     num_levels = len(cls_scores)
    # 
    #     device = cls_scores[0].device
    #     featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    #     mlvl_anchors = self.anchor_generator.grid_anchors(
    #         featmap_sizes, device=device)
    # 
    #     # convert anchors to rf
    #     result_list = []
    #     for img_id, _ in enumerate(img_metas):
    #         offset_list = [
    #             offsets[i][img_id].detach() for i in range(num_levels)
    #         ]
    #         cls_score_list = [
    #             cls_scores[i][img_id].detach() for i in range(num_levels)
    #         ]
    #         bbox_pred_list = [
    #             bbox_preds[i][img_id].detach() for i in range(num_levels)
    #         ]
    #         img_shape = img_metas[img_id]['img_shape']
    #         scale_factor = img_metas[img_id]['scale_factor']
    #         if with_nms:
    #             # some heads don't support with_nms argument
    #             proposals = self._get_bboxes_single(cls_score_list,
    #                                                 bbox_pred_list,
    #                                                 mlvl_anchors, img_shape,
    #                                                 scale_factor, cfg, rescale)
    #         else:
    #             proposals = self._get_bboxes_single(cls_score_list,
    #                                                 bbox_pred_list,
    #                                                 mlvl_anchors, img_shape,
    #                                                 scale_factor, cfg, rescale,
    #                                                 with_nms)
    #         result_list.append(proposals)
    #     return result_list
    # 
    # def _get_bboxes_single(self,
    #                         cls_score_list,
    #                         bbox_pred_list,
    #                         mlvl_anchors,
    #                         img_shape,
    #                         scale_factor,
    #                         cfg,
    #                         rescale=False,
    #                         with_nms=True):
    #     """Transform outputs for a single batch item into bbox predictions.
    # 
    #         Args:
    #             cls_score_list (list[Tensor]): Box scores for a single scale level
    #                 Has shape (num_anchors * num_classes, H, W).
    #             bbox_pred_list (list[Tensor]): Box energies / deltas for a single
    #                 scale level with shape (num_anchors * 4, H, W).
    #             mlvl_anchors (list[Tensor]): Box reference for a single scale level
    #                 with shape (num_total_anchors, 4).
    #             img_shape (tuple[int]): Shape of the input image,
    #                 (height, width, 3).
    #             scale_factor (ndarray): Scale factor of the image arange as
    #                 (w_scale, h_scale, w_scale, h_scale).
    #             cfg (mmcv.Config): Test / postprocessing configuration,
    #                 if None, test_cfg would be used.
    #             rescale (bool): If True, return boxes in original image space.
    #                 Default: False.
    #             with_nms (bool): If True, do nms before return boxes.
    #                 Default: True.
    # 
    #         Returns:
    #             Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
    #                 are bounding box positions (cx, cy, w, h, a) and the
    #                 6-th column is a score between 0 and 1.
    #     """
    #     cfg = self.test_cfg if cfg is None else cfg
    #     assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
    #     mlvl_bboxes = []
    #     mlvl_scores = []
    #     for cls_score, bbox_pred, anchors in zip(cls_score_list,
    #                                                  bbox_pred_list, mlvl_anchors):
    #         assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
    #         cls_score = cls_score.permute(1, 2,
    #                                         0).reshape(-1, self.cls_out_channels)
    #         if self.use_sigmoid_cls:
    #             scores = cls_score.sigmoid()
    #         else:
    #             scores = cls_score.softmax(-1)
    #         bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
    #         nms_pre = cfg.get('nms_pre', -1)
    #         if nms_pre > 0 and scores.shape[0] > nms_pre:
    #             # Get maximum scores for foreground classes.
    #             if self.use_sigmoid_cls:
    #                 max_scores, _ = scores.max(dim=1)
    #             else:
    #                 # remind that we set FG labels to [0, num_class-1]
    #                 # since mmdet v2.0
    #                 # BG cat_id: num_class
    #                 max_scores, _ = scores[:, :-1].max(dim=1)
    #             _, topk_inds = max_scores.topk(nms_pre)
    #             anchors = anchors[topk_inds, :]
    #             bbox_pred = bbox_pred[topk_inds, :]
    #             scores = scores[topk_inds, :]
    #         bboxes = self.bbox_coder.decode(
    #             anchors, bbox_pred, max_shape=img_shape)
    #         mlvl_bboxes.append(bboxes)
    #         mlvl_scores.append(scores)
    #     mlvl_bboxes = torch.cat(mlvl_bboxes)
    #     if rescale:
    #         # angle should not be rescaled
    #         mlvl_bboxes[:, :4] = mlvl_bboxes[:, :4] / mlvl_bboxes.new_tensor(
    #             scale_factor)
    #     mlvl_scores = torch.cat(mlvl_scores)
    #     if self.use_sigmoid_cls:
    #         # Add a dummy background class to the backend when using sigmoid
    #         # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
    #         # BG cat_id: num_class
    #         padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
    #         mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
    # 
    #     if with_nms:
    #         det_bboxes, det_labels = multiclass_nms(
    #             mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms,
    #             cfg.max_per_img)
    #         return det_bboxes, det_labels
    #     else:
    #         return mlvl_bboxes, mlvl_scores
    # 
    # 
    # 
    # 
    #     # if self.rf_based == True:
    #     #     mlvl_anchors = self.anchor_generator.anchor2rf(
    #     #         mlvl_anchors, decay=self.decay, device=device)
    #     #
    #     # mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
    #     # mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]
    #     #
    #     # if torch.onnx.is_in_onnx_export():
    #     #     assert len(
    #     #         img_metas
    #     #     ) == 1, 'Only support one input image while in exporting to ONNX'
    #     #     img_shapes = img_metas[0]['img_shape_for_onnx']
    #     # else:
    #     #     img_shapes = [
    #     #         img_metas[i]['img_shape']
    #     #         for i in range(cls_scores[0].shape[0])
    #     #     ]
    #     # scale_factors = [
    #     #     img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
    #     # ]
    #     #
    #     # if with_nms:
    #     #     # some heads don't support with_nms argument
    #     #     result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds,
    #     #                                    mlvl_anchors, img_shapes,
    #     #                                    scale_factors, cfg, rescale)
    #     # else:
    #     #     result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds,
    #     #                                    mlvl_anchors, img_shapes,
    #     #                                    scale_factors, cfg, rescale,
    #     #                                    with_nms)
    #     # return result_list
    # 
    # def _get_bboxes(self,
    #                 mlvl_cls_scores,
    #                 mlvl_bbox_preds,
    #                 mlvl_anchors,
    #                 img_shapes,
    #                 scale_factors,
    #                 cfg,
    #                 rescale=False,
    #                 with_nms=True):
    #     """Transform outputs for a batch item into bbox predictions.
    # 
    #     Args:
    #         mlvl_cls_scores (list[Tensor]): Each element in the list is
    #             the scores of bboxes of single level in the feature pyramid,
    #             has shape (N, num_anchors * num_classes, H, W).
    #         mlvl_bbox_preds (list[Tensor]):  Each element in the list is the
    #             bboxes predictions of single level in the feature pyramid,
    #             has shape (N, num_anchors * 4, H, W).
    #         mlvl_anchors (list[Tensor]): Each element in the list is
    #             the anchors of single level in feature pyramid, has shape
    #             (num_anchors, 4).
    #         img_shapes (list[tuple[int]]): Each tuple in the list represent
    #             the shape(height, width, 3) of single image in the batch.
    #         scale_factors (list[ndarray]): Scale factor of the batch
    #             image arange as list[(w_scale, h_scale, w_scale, h_scale)].
    #         cfg (mmcv.Config): Test / postprocessing configuration,
    #             if None, test_cfg would be used.
    #         rescale (bool): If True, return boxes in original image space.
    #             Default: False.
    #         with_nms (bool): If True, do nms before return boxes.
    #             Default: True.
    # 
    #     Returns:
    #         list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
    #             The first item is an (n, 5) tensor, where 5 represent
    #             (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
    #             The shape of the second tensor in the tuple is (n,), and
    #             each element represents the class label of the corresponding
    #             box.
    #     """
    #     cfg = self.test_cfg if cfg is None else cfg
    #     assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(
    #         mlvl_anchors)
    #     batch_size = mlvl_cls_scores[0].shape[0]
    #     # convert to tensor to keep tracing
    #     nms_pre_tensor = torch.tensor(
    #         cfg.get('nms_pre', -1),
    #         device=mlvl_cls_scores[0].device,
    #         dtype=torch.long)
    # 
    #     mlvl_bboxes = []
    #     mlvl_scores = []
    #     for cls_score, bbox_pred, anchors in zip(mlvl_cls_scores,
    #                                              mlvl_bbox_preds,
    #                                              mlvl_anchors):
    #         assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
    #         cls_score = cls_score.permute(0, 2, 3,
    #                                       1).reshape(batch_size, -1,
    #                                                  self.cls_out_channels)
    #         if self.use_sigmoid_cls:
    #             scores = cls_score.sigmoid()
    #         else:
    #             scores = cls_score.softmax(-1)
    #         bbox_pred = bbox_pred.permute(0, 2, 3,
    #                                       1).reshape(batch_size, -1, 4)
    #         anchors = anchors.expand_as(bbox_pred)
    #         # Always keep topk op for dynamic input in onnx
    #         from mmdet.core.export import get_k_for_topk
    #         nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
    #         if nms_pre > 0:
    #             # Get maximum scores for foreground classes.
    #             if self.use_sigmoid_cls:
    #                 max_scores, _ = scores.max(-1)
    #             else:
    #                 # remind that we set FG labels to [0, num_class-1]
    #                 # since mmdet v2.0
    #                 # BG cat_id: num_class
    #                 max_scores, _ = scores[..., :-1].max(-1)
    # 
    #             _, topk_inds = max_scores.topk(nms_pre)
    #             batch_inds = torch.arange(batch_size).view(
    #                 -1, 1).expand_as(topk_inds)
    #             anchors = anchors[batch_inds, topk_inds, :]
    #             bbox_pred = bbox_pred[batch_inds, topk_inds, :]
    #             scores = scores[batch_inds, topk_inds, :]
    # 
    #         bboxes = self.bbox_coder.decode(
    #             anchors, bbox_pred, max_shape=img_shapes)
    #         mlvl_bboxes.append(bboxes)
    #         mlvl_scores.append(scores)
    # 
    #     batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
    #     if rescale:
    #         batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
    #             scale_factors).unsqueeze(1)
    #     batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
    # 
    #     # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
    #     if torch.onnx.is_in_onnx_export() and with_nms:
    #         from mmdet.core.export import add_dummy_nms_for_onnx
    #         # ignore background class
    #         if not self.use_sigmoid_cls:
    #             num_classes = batch_mlvl_scores.shape[2] - 1
    #             batch_mlvl_scores = batch_mlvl_scores[..., :num_classes]
    #         max_output_boxes_per_class = cfg.nms.get(
    #             'max_output_boxes_per_class', 200)
    #         iou_threshold = cfg.nms.get('iou_threshold', 0.5)
    #         score_threshold = cfg.score_thr
    #         nms_pre = cfg.get('deploy_nms_pre', -1)
    #         return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
    #                                       max_output_boxes_per_class,
    #                                       iou_threshold, score_threshold,
    #                                       nms_pre, cfg.max_per_img)
    #     if self.use_sigmoid_cls:
    #         # Add a dummy background class to the backend when using sigmoid
    #         # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
    #         # BG cat_id: num_class
    #         padding = batch_mlvl_scores.new_zeros(batch_size,
    #                                               batch_mlvl_scores.shape[1],
    #                                               1)
    #         batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)
    # 
    #     if with_nms:
    #         det_results = []
    #         for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes,
    #                                               batch_mlvl_scores):
    #             det_bbox, det_label = multiclass_nms(mlvl_bboxes, mlvl_scores,
    #                                                  cfg.score_thr, cfg.nms,
    #                                                  cfg.max_per_img)
    #             det_results.append(tuple([det_bbox, det_label]))
    #     else:
    #         det_results = [
    #             tuple(mlvl_bs)
    #             for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)
    #         ]
    #     return det_results


    







