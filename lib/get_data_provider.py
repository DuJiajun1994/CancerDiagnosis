from data_providers.cancer_annotated import CancerAnnotated
from data_providers.cancer_not_annotated import CancerNotAnnotated


def get_data_provider(data_name, cfg):
    data_provider = None
    if data_name == 'cancer_annotated':
        data_provider = CancerAnnotated(cfg)
    elif data_name == 'cancer_not_annotated':
        data_provider = CancerNotAnnotated(cfg)
    assert data_provider is not None, \
        'data provider {} is not existed'.format(data_name)
    return data_provider