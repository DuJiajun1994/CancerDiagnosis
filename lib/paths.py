import os
from easydict import EasyDict

paths = EasyDict()
paths.root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
paths.data_path = os.path.join(paths.root_path, 'data')
paths.output_path = os.path.join(paths.root_path, 'output')
paths.cfg_path = os.path.join(paths.root_path, 'cfgs')

