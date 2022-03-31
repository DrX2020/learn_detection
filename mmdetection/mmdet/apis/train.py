# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)

from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import find_latest_checkpoint, get_root_logger


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(log_level=cfg.log_level)

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

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # no 'imgs_per_gpu' in cfg.data so this condition could be neglected
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    # runner = dict(type='EpochBasedRunner', max_epochs=...)
    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']
    
    # ******************************build data loaders******************************
    # build_dataloader() returns a PyTorch dataloader, i.e. an object of class Dataloader. 
    # The class of Dataloader Combines a dataset and a sampler, and provides an iterable over the given dataset
    data_loaders = [
        build_dataloader(
            # ds: object of a dataset class(e.g. CocoDataset) 
            ds,
            # cfg.data.samples_per_gpu: int
            cfg.data.samples_per_gpu,
            # cfg.data.workers_per_gpu: int
            cfg.data.workers_per_gpu,
            # `num_gpus` will be ignored if distributed
            # num_gpus: int
            num_gpus=len(cfg.gpu_ids),
            # if dist_train.sh is used, args.launcher is not None, dist is True
            # if original train.py is used, args.launcher is None, dist is False
            dist=distributed,
            # cfg.seed is not None, it is defined in mmdetection/tools/train.py
            seed=cfg.seed,
            # runner_type is 'EpochBasedRunner' by default
            runner_type=runner_type,
            # no 'persistent_workers' in cfg.data
            persistent_workers=cfg.data.get('persistent_workers', False))
        for ds in dataset
    ]

    # ******************************put model on gpus******************************
    # dist_train.sh is used
    # model.cuda() puts model on gpus
    # MMDistributedDataParallel(model.cuda(),...) wraps the model so that distributed multi-gpu can be applied
    # According to ../datasets/builder.py,
    # when model is :obj:`DistributedDataParallel`, `batch_size` of :obj:`dataloader` is the number of training samples on each GPU,
    # batch_size = samples_per_gpu.
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    # Original train.py is used
    # model.cuda(cfg.gpu_ids[0]) puts model on gpus
    # MMDataParallel(model.cuda(cfg.gpu_ids[0]),...) wraps the model so that multi-gpu can be applied 
    # (multi-gpu? Is that so? model is only put to cfg.gpu_ids[0])
    # According to ../datasets/builder.py, when model is obj:`DataParallel`, the batch size is samples on all the GPUS,
    # batch_size = num_gpus * samples_per_gpu
    else:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    # ******************************build optimizer******************************
    # e.g. cfg.optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
    optimizer = build_optimizer(model, cfg.optimizer)

    # ******************************build runner******************************
    # by default the 'if... else...' below should be neglected
    # runner is in cfg, e.g. runner = dict(type='EpochBasedRunner', max_epochs=3)
    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    # total_epochs is not in cfg
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    # e.g. cfg.runner = dict(type='EpochBasedRunner', max_epochs=3)
    # 4 steps for using Runner:
    # 1) initialize the Runner object
    # 2) register hooks
    # 3) call resume or load_checkpoint to get model parameters
    # 4) run the given workflow    
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    # register necessary training hooks
    # register_training_hooks() originally defined in mmcv/runner/base_runner.py
    
    # As long as the registered Hook object realize one or some functions (for a certain period, like 'after_train_iter period'), 
    # the function(s) will be called once Runner reach that period.
    # For example, in mmcv/runner/hooks/optimizer.py, OptimizerHook get after_train_iter period, 
    # so in train() in mmcv/runner/epoch_based_runner.py, clip_grads() defined in OptimizerHook is called every time self.call_hook('after_val_iter') is reached.    
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    # resume or load checkpoints
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    
    # when entering the corresponding workflow, 
    # train() or val() in .../site-packages/mmcv/runner/epoch_based_runner.py will be called,
    # in which self.call_hook('period') will be called, run_iter() will also  be called,
    # self.call_hook('period') calls all registered hooks in the corresponding period,
    # e.g. OptimizerHook will be called 'after_train_iter', as defined in mmcv/runner/hooks/optimizer.py
    # in run_iter(), train_step() or val_step() will be called, actually calling train_step() or val_step() of model(class MMDataParallel), 
    # again actually calling train_step() or val_step() of class BaseDetector
    # train_step() or val_step() of model are defined in mmcv/parallel/data_parallel.py/MMDataParallel
    # train_step() or val_step() of class BaseDetector are defined in mmdet/models/detectors/base.py/BaseDetector
    # forward() in mmdet/models/detectors/base.py/BaseDetector either calls forward_train() to return loss or calls forward_test() to return predict results
    # forward_train() defined in mmdet/models/detectors/two_stage.py/TwoStageDetector or mmdet/models/detectors/single_stage.py/SingleStageDetector
    runner.run(data_loaders, cfg.workflow)
