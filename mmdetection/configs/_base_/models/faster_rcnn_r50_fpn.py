# model settings
model = dict(
    type='FasterRCNN',
    backbone=dict(
        # /home/user/xiongdengrui/learn_detection/mmdetection/mmdet/models/backbones/resnet.py
        type='ResNet',
        depth=50,
        # ResNet is comprised of a stem and 4 stages
        num_stages=4,
        # the output feature maps of all 4 stages 
        # ([strides, #channels]: C2[4, 256], C3[8, 512], C4[16, 1024], C5[32, 2048], index from 0 to 3) 
        # are all needed 
        out_indices=(0, 1, 2, 3),
        # freeze stem and the first stage
        frozen_stages=1,
        # type of normalization operator and whether the parameters need to be updated
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        # PyTorch way to construct Bottleneck module, different from that of Caffe
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        # /home/user/xiongdengrui/learn_detection/mmdetection/mmdet/models/necks/fpn.py
        # generates FPN outputs P2 - P5 using ResNet feature maps C2 - C5 (channel [256, 512, 1024, 2048])
        # generates FPN output P6 with C5
        # 5 outputs of FPN are P2 - P6 (channel 256)
        # P2 stride: 4, P3 stride: 8, P4 stride: 16, P5 stride: 32, P6 stride: 64
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        # /home/user/xiongdengrui/learn_detection/mmdetection/mmdet/models/dense_heads/rpn_head.py
        # 5 outputs of FPN become 5 inputs of RPN Head
        # 5 inputs of RPN Head share one single set of classification branch parameters 
        # 5 inputs of RPN Head share one single set of regression branch parameters 
        type='RPNHead',
        # #channels of RPN output feature maps
        in_channels=256,
        # #channels of middle feature maps
        feat_channels=256,
        anchor_generator=dict(
            # /home/user/xiongdengrui/learn_detection/mmdetection/mmdet/core/anchor/anchor_generator.py
            type='AnchorGenerator',
            # base scale of anchors, if input image is ixi
            # 8x8 anchor in feature map of stride 4 (i/4)x(i/4) correspond to 32x32 anchor in input image, aiming at small objects
            # 8x8 anchor in feature map of stride 64 (i/64)x(i/64) correspond to 512x512 anchor in input image, aiming at large objects
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            # bbox encoder/decoder balances losses of classification and regression, and balances losses of 4 predictions of bbox branch
            # bbox encoder/decoder introduces anchor information to promote convergence
            # /home/user/xiongdengrui/learn_detection/mmdetection/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        # positive and negative anchors contribute to loss_cls
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # only positive anchors contribute to loss_bbox
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        # /home/user/xiongdengrui/learn_detection/mmdetection/mmdet/models/roi_heads/standard_roi_head.py
        # simplest base roi head including one bbox head and one mask head
        # input shape of R-CNN head: (batch size, #proposals after NMS per img, 4), 4 indicates the coordinates of RoI
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            # RoI Align input size: (batch size, #proposals after NMS per img, 4)
            # RoI Align output size: (batch, #proposals after NMS per img, 256, roi_feat_size, roi_feat_size)
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            # stride 64 is not used
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            # input size of Shared2FCBBoxHead: (batch*(#proposals after NMS per img), 256*roi_feat_size*roi_feat_size), resized from RoI Align output size
            # output size of Shared2FCBBoxHead: (batch*(#proposals after NMS per img), 1024)
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            # output size of Shared2FCBBoxHead is delivered to classification branch and regression branch
            # classification branch output size: (batch*(#proposals after NMS per img), (num_classes+1))
            # regression branch output size: (batch*(#proposals after NMS per img), 4*num_classes+1)
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        # rpn corresponds to train_cfg in RetinaNet, rpn_proposal corresponds to test_cfg in RetinaNet
        rpn=dict(
            assigner=dict(
                # # /home/user/xiongdengrui/learn_detection/mmdetection/mmdet/core/bbox/assigners/max_iou_assigner.py
                type='MaxIoUAssigner',
                # an anchor that has the highest IOU with a GT box will be assigned positive
                # an anchor that has an IoU overlap higher than 0.7 with one ground-truth box will be assigned positive
                # an anchor that has an IoU overlap lower than 0.3 with any ground-truth box will be assigned negative
                # anchors that are neither assigned positive nor assigned negative will not contribute to the training loss (no such anchors if neg_iou_thr = min_pos_iou)
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                # minimum iou for a bbox to be considered as a positive bbox
                min_pos_iou=0.3,
                # match low-quality anchor to a GT that has no positive anchor assignment
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                # /home/user/xiongdengrui/learn_detection/mmdetection/mmdet/core/bbox/samplers/random_sampler.py
                type='RandomSampler',
                # training samples
                num=256,
                # ratio of positive samples
                pos_fraction=0.5,
                # upper bound number of negative and positive samples, defaults to -1
                # if num=256, pos_fraction=0.5 but positive sample is less than 128, 
                # setting neg_pos_ub=-1 will get more than 128 negative samples to get all 256 samples
                # setting neg_pos_ub to a positive number p will sample negative samples that are p times of positive
                neg_pos_ub=-1,
                # whether to add GT as proposals to get high-quality positive sample
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            # max number of proposals per stride (5 strides per image in total) before NMS, sorted by classification score of this stride
            nms_pre=2000,
            # max number of proposals of the final RPN prediction per image
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            # minimum bbox size, bboxes smaller than this will be filtered out
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                # an RoI that has the highest IOU with a GT box will be assigned to that class
                # an RoI that has an IoU overlap higher than 0.5 with one ground-truth box will be assigned to that class
                # an RoI that has an IoU overlap lower than 0.5 with any ground-truth box will be assigned negative
                # RoIs that are neither assigned a class nor assigned negative will not contribute to the training loss (no such RoIs if neg_iou_thr = min_pos_iou)
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                # /home/user/xiongdengrui/learn_detection/mmdetection/mmdet/core/bbox/samplers/random_sampler.py
                type='RandomSampler',
                # training samples
                num=512,
                # ratio of positive samples
                pos_fraction=0.25,
                # upper bound number of negative and positive samples, defaults to -1
                # if num=512, pos_fraction=0.25 but positive sample is less than 128, 
                # setting neg_pos_ub=-1 will get more than 384 negative samples to get all 512 samples
                # setting neg_pos_ub to a positive number p will sample negative samples that are p times of positive
                neg_pos_ub=-1,
                # whether to add GT as proposals to get high-quality positive sample
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            # max number of proposals per stride (5 strides per image in total) before NMS, sorted by classification score of this stride
            nms_pre=1000,
            # max number of proposals of the final RPN prediction per image
            max_per_img=1000,
            # NMS method and threshold
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            # score threshold, predictions with lower score will be discarded
            score_thr=0.05,
            # NMS method and threshold
            nms=dict(type='nms', iou_threshold=0.5),
            # max number of proposals of the final RPN prediction per image
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
