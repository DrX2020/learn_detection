1. 修改配置文件里dataset部分。
2. mmdet/dataset里增加相应数据集的.py文件。
3. mmdet/core/evaluation/mean_ap.py定制。
4. mmdet/dataset/____init____.py添加对应数据集内容，注意上下都要，修改两个地方。
5. mmdet/core/evaluation/class_names.py添加对应数据集内容，注意除了添加一个函数的定义，字典dataset_aliases也要改，字典中键+"_classes"应为添加的函数名。
6. 模型配置中，num_classes要改。
