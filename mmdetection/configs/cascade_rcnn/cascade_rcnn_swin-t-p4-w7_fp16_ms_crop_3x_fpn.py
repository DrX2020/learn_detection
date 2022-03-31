_base_ = './cascade_rcnn_swin-t-p4-w7_ms_crop_3x_fpn.py'
# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))
