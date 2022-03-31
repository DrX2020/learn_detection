python ./tools/test.py \
    ./work_dir/20220322162839/retinanet_swin-t-p4-w7_fpn_1x_coco/draw_predict_and_gt_boxes_on_val.py \
    ./work_dir/20220322162839/retinanet_swin-t-p4-w7_fpn_1x_coco/best_ave_acc_epoch_30.pth \
    --gpu-ids 0 \
    --out ./work_dir/20220322162839/retinanet_swin-t-p4-w7_fpn_1x_coco/result.pkl \
    --show-dir ./work_dir/20220322162839/retinanet_swin-t-p4-w7_fpn_1x_coco/trainval_result