def dataset_info(dataset, meta=None):
    '''
    Return metadata associated with certain datasets.
    '''

    info = {
        'CelebA': {
            'orig_res': (178, 218),
            'img_dir': 'data/CelebA/img_align_celeba/',
            'target_res': (224, 224),
            'transform': 'base'
        },
        'WaterBirds': {
            'orig_res': (224, 224),
            'img_dir': 'data/WaterBirds/',
            'target_res': (224, 224),
            'transform': 'resize_center_crop',
            'resize_scale': 256.0/224.0,
        },
        'Camelyon17': {
            'orig_res': (96, 96),
            'img_dir': 'data/Camelyon17/patches/',
            'target_res': (96, 96),
            'transform': 'base',
        }
    }
    if meta is None: return info[dataset]
    return info[dataset][meta]
