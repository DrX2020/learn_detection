# ************************************retinanet_swin-t-p4-w7_fpn_1x_coco***************************************

time=$(date "+%Y%m%d%H%M%S")
work_folder=./work_dir/${time}
config_folder='./configs/swin/'

for method in retinanet_swin-t-p4-w7_fpn_1x_coco;
do
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh \
    ${config_folder}/${method}.py \
    2 \
    --work-dir ./${work_folder}/${method}/
done

# ************************************retinanet_swin-t-p4-w7_fpn_1x_coco***************************************

# ************************************yolox***************************************

# time=$(date "+%Y%m%d%H%M%S")
# work_folder=./work_dir/${time}
# config_folder='./configs/yolox/'

# for method in yolox_x_8x8_300e_coco;
# do
# CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh \
#     ${config_folder}/${method}.py \
#     2 \
#     --work-dir ./${work_folder}/${method}/
# done

# ************************************yolox***************************************