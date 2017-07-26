from easydict import EasyDict
import os
import yaml


def get_config(cfg_name):
    cfg_file = os.path.join('cfgs', '{}.yml'.format(cfg_name))
    assert os.path.exists(cfg_file), \
        'config file {} is not existed'.format(cfg_file)
    with open(cfg_file, 'r') as f:
        cfg = EasyDict(yaml.load(f))
        cfg.snapshot_step = 1000
        return cfg