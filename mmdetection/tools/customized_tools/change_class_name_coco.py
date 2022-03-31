import json
import numpy as np
from tqdm import tqdm

original_file_path = "/home/user/xiongdengrui/Split_Dataset_Combine/trainval.json"
destination_file_path = "/home/user/xiongdengrui/Split_Dataset_Combine/trainval_combine.json"

# open file, get load_ori(_io.TextIOWrapper type)
with open(original_file_path,'r') as load_ori:
    # get load_ori_json(dict corresponding to load_ori)
    load_ori_json = json.load(load_ori)
    print(load_ori_json.keys())
    
data_dict = {}
data_dict['images'] = []
data_dict['type'] = "instances"
data_dict['annotations'] = []
data_dict['categories'] = [
        {
            "id": 1,
            "name": "dirt",
            "supercategory": "none"
        },
        {
            "id": 2,
            "name": "scratch",
            "supercategory": "none"
        },
        {
            "id": 3,
            "name": "pit",
            "supercategory": "none"
        }
    ]

print("---------------------load images----------------------")
for image in tqdm(load_ori_json["images"]):
    data_dict['images'].append(image)

print("---------------------load and process annotations----------------------")    
for annotation in tqdm(load_ori_json["annotations"]):
    if annotation["category_id"] == 3:
        annotation["category_id"] = 2
    elif annotation["category_id"] == 4:
        annotation["category_id"] = 3
    data_dict['annotations'].append(annotation)

# open file, get load_des(_io.TextIOWrapper type)  
with open(destination_file_path, 'w') as load_des:
    json.dump(data_dict, load_des)
    

# type of load_json: dict
# print("load_json:")    
# print(type(load_json), load_json)

# type of load_f: _io.TextIOWrapper
# print("load_f:")    
# print(type(load_f), load_f)