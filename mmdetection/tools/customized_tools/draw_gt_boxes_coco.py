import os
import xml.dom.minidom
import cv2 as cv
import json

# 图片文件所在目录 
ImgPath = '/home/user/xiongdengrui/learn_detection/mmdetection/work_dir/20220322162839/retinanet_swin-t-p4-w7_fpn_1x_coco/trainval_result/' 
# json文件地址
json_file_path = '/home/user/xiongdengrui/learn_detection/dataset_ship/annotations/instances_all.json'  
# 图片文件保存的地址，需要先建文件夹
save_path = '/home/user/xiongdengrui/learn_detection/mmdetection/work_dir/20220322162839/retinanet_swin-t-p4-w7_fpn_1x_coco/trainval_result_with_gt/' 

# open file, get load_ori(_io.TextIOWrapper type)
with open(json_file_path,'r') as load_ori:
    # get load_ori_json(dict corresponding to load_ori)
    load_ori_json = json.load(load_ori)
    print(load_ori_json.keys())

# 得到乱序的以文件名字符串为元素的列表
imagelist = os.listdir(ImgPath)
# 简化处理，先用少量图片
imagelist = imagelist[0: 10]

# GT框的颜色，元素个数需和类数一致
gt_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

# 遍历每张图，对其进行画框与保存
for image in imagelist:
    # 分离文件名和扩展名，得到image_id
    image_pre, ext = os.path.splitext(image)
    # 筛选出以.jpg结尾的文件，用于文件夹含有其它格式的时候
    if ext == '.jpg':       
        # 图片的完整路径
        imgfile = ImgPath + image
        # 读取图片
        img = cv.imread(imgfile)
        # 图片id
        image_id = int(image_pre)
        print(image_id)
        # 存放该图片所有标注框的列表
        image_bboxes = []
        # 得到该图片对应的所有的标注框，放入image_bboxes
        for annotation in load_ori_json["annotations"]:
            if annotation["image_id"] == image_id:
                x1 = annotation["bbox"][0]
                x2 = x1 + annotation["bbox"][2]
                y1 = annotation["bbox"][1]
                y2 = y1 + annotation["bbox"][3]
                # image_bboxes.append([x1, y1, x2, y2])
                for catogory in load_ori_json["categories"]:
                    if catogory["id"] == annotation["category_id"]:
                        image_bboxes.append([x1, y1, x2, y2, catogory["name"]])
        # 画框与写标签
        for image_bbox in image_bboxes:
            # print(image_bbox)
            cv.rectangle(img, (image_bbox[0], image_bbox[1]), (int(image_bbox[2]), int(image_bbox[3])), gt_colors[annotation["category_id"]-1], thickness=1)
            cv.putText(img, image_bbox[4], (image_bbox[0], image_bbox[1]), cv.FONT_HERSHEY_COMPLEX, 0.4, gt_colors[annotation["category_id"]-1], thickness=1)
        # 保存
        cv.imwrite(save_path+'/'+image_pre+'.jpg', img)