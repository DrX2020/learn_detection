# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger, setup_multi_processes


def parse_args():
    # Object for parsing command line strings into Python objects, it allows for user-friendly command api
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    # read arguments in command, e.g. config:
    #  /home/user/xiongdengrui/Det_Models/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_annotated.py \
    # gpu-ids:
    #  --gpu-ids 1 \
    # work-dir:
    #  --work-dir /home/user/xiongdengrui/work_dirs/20220226/1_faster_r50_learn_code
    args = parse_args()

    # train config file path, read contents in train config file and load them into cfg
    # Config.fromfile parses config file and returns an object of class Config itself
    # e.g.
    # cfg:
    # "Config [path: /home/kchen/projects/mmcv/tests/data/config/a.py]: "
    # "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    # no distributed
    if args.launcher == 'none':
        distributed = False
    # args.launcher is not 'none' then using dist_train.sh so in this case distributed = True,
    # `batch_size` of :obj:`dataloader` is the number of training samples on each GPU, 
    # according to ../mmdet/datasets/builder.py.
    # distributed
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config, in other word, save config file under work_dir
    # osp.basename(args.config) get basename of args.config, 
    # e.g. get c.py from /a/b/c.py
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log file path
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # in default_runtime, log_level = 'INFO'
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    # Environment info, Distributed training, Config, Set random seed, deterministic can all be found in the saved log file
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    # set random seeds for python, numpy, torch
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    # ******************************build the model******************************
    # def build_detector(cfg, train_cfg=None, test_cfg=None):
    # """Build detector."""
    # if train_cfg is not None or test_cfg is not None:
    #     warnings.warn(
    #         'train_cfg and test_cfg is deprecated, '
    #         'please specify them in model', UserWarning)
    # assert cfg.get('train_cfg') is None or train_cfg is None, \
    #     'train_cfg specified in both outer field and model field '
    # assert cfg.get('test_cfg') is None or test_cfg is None, \
    #     'test_cfg specified in both outer field and model field '
    # return DETECTORS.build(
    #     cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
    
    # build_detector() receives 3 dicts as parameters and returns an object of registered class in DETECTORS
    # For example, build_detector() returns an object of FasterRCNN(registered in ../models/detectors/faster_rcnn.py) in DETECTORS
    # The most bottom function of build_detector() is build_from_cfg(cfg, registry, default_args=None), parameter delivery:
    # cfg.model -> cfg, cfg.model is a dict defined in python config file right after "model ="
    # DETECTORS -> registry
    # dict(train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')) -> default_args
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # ******************************build the dataset******************************
    datasets = [build_dataset(cfg.data.train)]
    # len(cfg.workflow) is 1 by default
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        # in default_runtime.py, checkpoint_config = dict(interval=100)
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    
    # ******************************train the detector******************************
    # model is an instantialized object of a class in DETECTORS
    # e.g. model is an instantialized object of class FasterRCNN
    
    # dataset is a list containing an instantialized object of a class in DATASETS
    # e.g. dataset is a list containing an instantialized object of class CocoDataset
    
    # cfg = Config.fromfile(args.config)
    # cfg is an object of class Config
    # e.g. 
    # cfg:
    # "Config [path: /home/kchen/projects/mmcv/tests/data/config/a.py]: "
    # "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
