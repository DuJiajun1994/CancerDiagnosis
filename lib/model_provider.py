from models.vgg16 import vgg16
from models.vgg16_fcn import vgg16_fcn

def get_model(model_name):
    if model_name == 'vgg16':
        return vgg16
    elif model_name == 'vgg16_fcn':
        return vgg16_fcn
    else:
        raise Exception('model {} is not existed'.format(model_name))
