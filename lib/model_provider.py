from models.vgg16 import vgg16


def get_model(model_name):
    if model_name == 'vgg16':
        return vgg16
    else:
        raise Exception('model {} is not existed'.format(model_name))
