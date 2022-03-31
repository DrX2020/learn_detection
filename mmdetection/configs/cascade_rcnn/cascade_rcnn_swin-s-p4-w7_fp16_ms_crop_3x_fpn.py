# _base_ = './cascade_rcnn_swin-t-p4-w7_fpn_fp16_ms_crop_3x.py'
_base_ = './cascade_rcnn_swin-t-p4-w7_fp16_ms_crop_3x_fpn.py'
pretrained = './configs/_base_/swin_ckpt/swin_small_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        depths=[2, 2, 18, 2],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))
