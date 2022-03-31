import numpy as np
from mmcv.visualization import imshow_bboxes
import matplotlib.pyplot as plt
from mmdet.core import build_anchor_generator

if __name__ == '__main__':
    anchor_generator_cfg = dict(
        type='AnchorGenerator',
        octave_base_scale=4,
        scales_per_octave=3,
        ratios=[0.5, 1.0, 2.0],
        strides=[8, 16, 32, 64, 128])
    anchor_generator = build_anchor_generator(anchor_generator_cfg)
    # 输出原图尺度上 anchor 坐标 xyxy 左上角格式
    # base_anchors 是长度为5的list，表示5个输出特征图，不同的特征图尺度相差的只是 strides
    # 故我们取 strides=8 的位置 anchor 可视化即可
    # base_anchors[i] is a tensor of torch.Size([9, 4]) 
    # that contains 4 coordinate values of all 9 anchors on (0, 0) in the ith feature map
    base_anchors = anchor_generator.base_anchors[0]
    
    # ***************************experiment 1****************************
    # print('anchor_generator.base_anchors:', type(anchor_generator.base_anchors), np.shape(anchor_generator.base_anchors))
    # for i in range (0, 5):
    #     base_anchors = anchor_generator.base_anchors[i]
    #     print(f'base_anchors[{i}]:', '\n', base_anchors, '\n', type(base_anchors), np.shape(base_anchors))
    
    # output:
    # anchor_generator.base_anchors: <class 'list'> (5,)
    # base_anchors[0]: 
    # tensor([[-22.6274, -11.3137,  22.6274,  11.3137],
    #         [-28.5088, -14.2544,  28.5088,  14.2544],
    #         [-35.9188, -17.9594,  35.9188,  17.9594],
    #         [-16.0000, -16.0000,  16.0000,  16.0000],
    #         [-20.1587, -20.1587,  20.1587,  20.1587],
    #         [-25.3984, -25.3984,  25.3984,  25.3984],
    #         [-11.3137, -22.6274,  11.3137,  22.6274],
    #         [-14.2544, -28.5088,  14.2544,  28.5088],
    #         [-17.9594, -35.9188,  17.9594,  35.9188]]) 
    # <class 'torch.Tensor'> torch.Size([9, 4])
    # base_anchors[1]: 
    # tensor([[-45.2548, -22.6274,  45.2548,  22.6274],
    #         [-57.0175, -28.5088,  57.0175,  28.5088],
    #         [-71.8376, -35.9188,  71.8376,  35.9188],
    #         [-32.0000, -32.0000,  32.0000,  32.0000],
    #         [-40.3175, -40.3175,  40.3175,  40.3175],
    #         [-50.7968, -50.7968,  50.7968,  50.7968],
    #         [-22.6274, -45.2548,  22.6274,  45.2548],
    #         [-28.5088, -57.0175,  28.5088,  57.0175],
    #         [-35.9188, -71.8376,  35.9188,  71.8376]]) 
    # <class 'torch.Tensor'> torch.Size([9, 4])
    # base_anchors[2]: 
    # tensor([[ -90.5097,  -45.2548,   90.5097,   45.2548],
    #         [-114.0350,  -57.0175,  114.0350,   57.0175],
    #         [-143.6751,  -71.8376,  143.6751,   71.8376],
    #         [ -64.0000,  -64.0000,   64.0000,   64.0000],
    #         [ -80.6349,  -80.6349,   80.6349,   80.6349],
    #         [-101.5937, -101.5937,  101.5937,  101.5937],
    #         [ -45.2548,  -90.5097,   45.2548,   90.5097],
    #         [ -57.0175, -114.0350,   57.0175,  114.0350],
    #         [ -71.8376, -143.6751,   71.8376,  143.6751]]) 
    # <class 'torch.Tensor'> torch.Size([9, 4])
    # base_anchors[3]: 
    # tensor([[-181.0193,  -90.5097,  181.0193,   90.5097],
    #         [-228.0701, -114.0350,  228.0701,  114.0350],
    #         [-287.3503, -143.6751,  287.3503,  143.6751],
    #         [-128.0000, -128.0000,  128.0000,  128.0000],
    #         [-161.2699, -161.2699,  161.2699,  161.2699],
    #         [-203.1873, -203.1873,  203.1873,  203.1873],
    #         [ -90.5097, -181.0193,   90.5097,  181.0193],
    #         [-114.0350, -228.0701,  114.0350,  228.0701],
    #         [-143.6751, -287.3503,  143.6751,  287.3503]]) 
    # <class 'torch.Tensor'> torch.Size([9, 4])
    # base_anchors[4]: 
    # tensor([[-362.0387, -181.0193,  362.0387,  181.0193],
    #         [-456.1401, -228.0701,  456.1401,  228.0701],
    #         [-574.7006, -287.3503,  574.7006,  287.3503],
    #         [-256.0000, -256.0000,  256.0000,  256.0000],
    #         [-322.5398, -322.5398,  322.5398,  322.5398],
    #         [-406.3747, -406.3747,  406.3747,  406.3747],
    #         [-181.0193, -362.0387,  181.0193,  362.0387],
    #         [-228.0701, -456.1401,  228.0701,  456.1401],
    #         [-287.3503, -574.7006,  287.3503,  574.7006]]) 
    # <class 'torch.Tensor'> torch.Size([9, 4])
    # ***************************experiment 1****************************

    h = 100
    w = 160
    # img: <class 'numpy.ndarray'> (100, 160, 3), all elements in this ndarray are 255
    img = np.ones([h, w, 3], np.uint8) * 255
    
    # ***************************experiment 2****************************
    # print('img:', type(img), np.shape(img))
    
    # output:
    # img: <class 'numpy.ndarray'> (100, 160, 3)
    # ***************************experiment 2****************************
    
    # ***************************experiment 3****************************
    base_anchors[:, 0::2] += w // 2
    base_anchors[:, 1::2] += h // 2
    print('base_anchors[:, 0::2] += w // 2:', '\n', base_anchors[:, 0::2])
    print('base_anchors[:, 1::2] += h // 2:', '\n', base_anchors[:, 1::2])
    
    # output:
    # base_anchors[:, 0::2] += w // 2: 
    # tensor([[ 57.3726, 102.6274],
    #         [ 51.4912, 108.5088],
    #         [ 44.0812, 115.9188],
    #         [ 64.0000,  96.0000],
    #         [ 59.8413, 100.1587],
    #         [ 54.6016, 105.3984],
    #         [ 68.6863,  91.3137],
    #         [ 65.7456,  94.2544],
    #         [ 62.0406,  97.9594]])
    # base_anchors[:, 1::2] += h // 2: 
    # tensor([[38.6863, 61.3137],
    #         [35.7456, 64.2544],
    #         [32.0406, 67.9594],
    #         [34.0000, 66.0000],
    #         [29.8413, 70.1587],
    #         [24.6016, 75.3984],
    #         [27.3726, 72.6274],
    #         [21.4912, 78.5088],
    #         [14.0812, 85.9188]])
    # ***************************experiment 3****************************
    
    # the (160, 100) image has no minus coordinate, 
    # the following 2 lines shift anchors to the middle of the (160, 100) image
    # base_anchors[:, 0::2] slices all the x coordinates, 
    # base_anchors[:, 1::2] slices all the y coordinates
    base_anchors[:, 0::2] += w // 2
    base_anchors[:, 1::2] += h // 2

    colors = ['green', 'red', 'blue']
    
    # ***************************experiment 4****************************
    # for i in range(3):
    #     base_anchor = base_anchors[i::3, :]
    #     print(base_anchor)
    
    # output:
    # tensor([[ 57.3726,  38.6863, 102.6274,  61.3137],
    #         [ 64.0000,  34.0000,  96.0000,  66.0000],
    #         [ 68.6863,  27.3726,  91.3137,  72.6274]])
    # tensor([[ 51.4912,  35.7456, 108.5088,  64.2544],
    #         [ 59.8413,  29.8413, 100.1587,  70.1587],
    #         [ 65.7456,  21.4912,  94.2544,  78.5088]])
    # tensor([[ 44.0812,  32.0406, 115.9188,  67.9594],
    #         [ 54.6016,  24.6016, 105.3984,  75.3984],
    #         [ 62.0406,  14.0812,  97.9594,  85.9188]])
    # ***************************experiment 4****************************    
    
    for i in range(3):
        base_anchor = base_anchors[i::3, :].cpu().numpy()
        imshow_bboxes(img, base_anchor, show=False, colors=colors[i])
    plt.grid()
    plt.imshow(img)
    plt.show()
    plt.savefig('visualize_anchors.jpg')