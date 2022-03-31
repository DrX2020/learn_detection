# %%
import os
import os.path as osp
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import random 
os.chdir("/home/user/sun_chen/Projects/ShipDetection")

def read_txt(txt_path):
    with open(txt_path,"r") as load_f:
        info_list = load_f.readlines()
        for i in range(len(info_list)):
            info_list[i] = info_list[i].strip("\n").split(" ")
            for j in range(len(info_list[i])):
                if j == 0:
                    info_list[i][j] = int(info_list[i][j])
                else:                
                    info_list[i][j] = float(info_list[i][j])*256
    return info_list        

def save_json(json_file,save_path):
    with open(save_path,"w") as load_f:
        json_file = json.dump(json_file,load_f,indent=4, separators=(',', ': '))
    print("save success!")

def read_json(json_path):
    with open(json_path,"r") as load_f:
        json_file = json.load(load_f)
    return json_file

#%%
## from txt file to json file(train_data)

train_img_folder = "./train_data"
file_list = [file_name for file_name in os.listdir(train_img_folder) if "txt" in file_name]

coco_json = {"images": [],
            "type": "instances",
            "annotations": [],
            "categories":  [{"supercategory": "none", "id": 1, "name": "ship"}]}

bbox_id = 0

for path in tqdm(file_list):
    
    bbox_list = read_txt(osp.join(train_img_folder,path))
    img_info = {"height":256,"width":256,"negative":0}
    img_info["file_name"] = path[:-3]+"jpg"
    img_info["id"] = int(path[:-4])
    coco_json["images"].append(img_info)
    for bbox in bbox_list:
        anno_info = {"iscrowd": 0,"negative": 0,"ignore": 0,"category_id": 1,"segmentation":[]}
        anno_info["image_id"] = int(path[:-4])
        anno_info["area"] = bbox[3]*bbox[4]
        anno_info["bbox"] = [int(bbox[1]-0.5*bbox[3]),int(bbox[2]-0.5*bbox[4]),bbox[3],bbox[4]]
        anno_info["id"] = bbox_id
        coco_json["annotations"].append(anno_info)
        bbox_id += 1
# save_json(coco_json,"./annotations/instances_all.json")
#%%
## split train val

def imgid2json(all_json_path,img_id_list,save_path):

    load_json = read_json(all_json_path)
                  
    category = load_json["categories"]

    train_dict = {"images": [],
                    "type": "instances",
                    "annotations": [],
                    "categories": category}

    for images in load_json["images"]:
        if images["id"] in img_id_list:
            train_dict["images"].append(images)
    for anno in load_json["annotations"]:
        if anno["image_id"] in img_id_list:
            train_dict["annotations"].append(anno)
    save_json(train_dict,save_path)

json_path = "./annotations/instances_all.json"
json_file = read_json(json_path)
img_list = [img["id"] for img in json_file["images"]]


split_point = 0.8
sample_num = int(split_point*len(img_list))
seed = 0
random.seed(seed)
train_img_id_list = random.sample(img_list,sample_num)
val_img_id_list = [img_id for img_id in img_list if img_id not in train_img_id_list]

imgid2json(json_path,train_img_id_list,"./annotations/train.json")
imgid2json(json_path,val_img_id_list,"./annotations/val.json")
#%%
### generate blank json file of test data for inference
test_img_folder = "./test_data"
file_list = [file_name for file_name in os.listdir(test_img_folder)]

coco_json = {"images": [],
            "type": "instances",
            "annotations": [],
            "categories":  [{"supercategory": "none", "id": 1, "name": "ship"}]}

for path in tqdm(file_list):    
    img_info = {"height":256,"width":256,"negative":0}
    img_info["file_name"] = path
    img_info["id"] = int(path[:-4])
    coco_json["images"].append(img_info)
    
# save_json(coco_json,"./annotations/test.json")

# %%
### img/bbox info collect
json_path = "./annotations/instances_all.json"

json_file = read_json(json_path)
img_num = len(json_file["images"])
anno_num = len(json_file["annotations"])
print(f"imgs: {img_num}\nbbox: {anno_num}")

fig = plt.figure()
w_h_ratio_list = [anno_info["bbox"][2]/anno_info["bbox"][3] for anno_info in json_file["annotations"]]
area_list = [anno_info["area"] for anno_info in json_file["annotations"]]

w_h_ratio_list.sort()
area_list.sort()
plt.subplot(211)
plt.hist(w_h_ratio_list,bins=100,facecolor="blue", edgecolor="black", alpha=0.7)
# 显示横轴标签
plt.xlabel("Ratio")
plt.xticks((0.5,1,2,5))

# 显示纵轴标签
plt.ylabel("Frequency")
# 显示图标题
plt.title("Ratio Distribution")
plt.subplot(212)
plt.hist(area_list, bins=100,facecolor="blue", edgecolor="black", alpha=0.7)
# 显示横轴标签
plt.xticks((1024,5000,10000,20000))
plt.xlabel("Area")
plt.ylabel("Frequency")

# 显示图标题
plt.title("Area Distribution")
plt.tight_layout()
plt.show()
# %%