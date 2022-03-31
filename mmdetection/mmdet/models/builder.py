# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)

BACKBONES = MODELS
NECKS = MODELS
ROI_EXTRACTORS = MODELS
SHARED_HEADS = MODELS
HEADS = MODELS
LOSSES = MODELS
DETECTORS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_roi_extractor(cfg):
    """Build roi extractor."""
    return ROI_EXTRACTORS.build(cfg)


def build_shared_head(cfg):
    """Build shared head."""
    return SHARED_HEADS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


# DETECTORS is an object of class Registry
# a specific detctor(e.g. Faster R-CNN) is registered in DETECTORS
# build_detector() returns an object of this specific detctor
def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    # usually train_cfg=None, test_cfg=None when using this function because train_cfg and test_cfg are defined explicitly in cfg
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    # build() originally defined in mmcv/utils/registry.py, 
    # registry_annotated.py is in the same folder with this .py file to facilitate the reference
 
    # definition of build():
    # def build(self, *args, **kwargs):
    #     return self.build_func(*args, **kwargs, registry=self)
    
    # DETECTORS.build() returns the object constructed according to (cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
    # (cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)) specifies the detector registered in DETECTORS to apply(e.g. Faster R-CNN) and detailed settings in it
    # cfg -> *args
    # default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg) -> **kwargs
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
