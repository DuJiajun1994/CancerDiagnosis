from config import Config
cfg_lib = {
    'cfg1': {
        'train_iters': 10000,
        'test_step': 1000,
        'learning_rate': 0.01,
        'batch_size': 100,
        'display_step': 100
    }
}


def parse_config(dict):
    cfg = Config()
    cfg.train_iters = dict['train_iters']
    cfg.test_step = dict['test_step']
    cfg.learning_rate = dict['learning_rate']
    cfg.batch_size = dict['batch_size']
    cfg.display_step = dict['display_step']
    return cfg


def get_config(cfg_name):
    if cfg_name in cfg_lib:
        cfg = parse_config(cfg_lib[cfg_name])
        return cfg
    else:
        raise Exception('config {} is not existed'.format(cfg_name))