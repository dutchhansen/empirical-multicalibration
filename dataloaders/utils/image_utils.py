import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os

_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]


## Pytorch dataset wrapper
class ImageDatasetWrapper(torch.utils.data.Dataset):
    '''
    Pytorch wrapper for image dataset.
    Allows easy loading and transformation of samples.

    Image transform code from WILDS:
    @inproceedings{wilds2021,
        title = {{WILDS}: A Benchmark of in-the-Wild Distribution Shifts},
        author = {Pang Wei Koh and Shiori Sagawa and Henrik Marklund and Sang Michael Xie and 
                    Marvin Zhang and Akshay Balsubramani and Weihua Hu and Michihiro Yasunaga and 
                    Richard Lanas Phillips and Irena Gao and Tony Lee and Etienne David and Ian Stavness and 
                    Wei Guo and Berton A. Earnshaw and Imran S. Haque and Sara Beery and Jure Leskovec and 
                    Anshul Kundaje and Emma Pierson and Sergey Levine and Chelsea Finn and Percy Liang},
        booktitle = {International Conference on Machine Learning (ICML)},
        year = {2021}
    }
    '''
    def __init__(self, X, y, config):
        '''
        Parameters:
            X: list of strings. File names of images.
            y: list of ints. Labels of images.
            config: dict. Configuration for image transformations.
            patches: bool. Whether the images are patches or not, 
                if True, then each x \in X is a list of file names.
        '''
        self.X = X
        if y is None: self.y = np.zeros(len(X), )
        else: self.y = y
        
        self.img_dir = config['img_dir']
        self.init_transform(transform_name=config['transform'], 
                             config=config)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # collect
        img_filename = os.path.join(
            self.img_dir, 
            self.X[idx]
        )
        
        # transform
        x = Image.open(img_filename).convert('RGB')
        x_t = self.transform(x)

        return x_t, self.y[idx]
    
    ## Transformations
    def init_transform(self, transform_name, config):
        '''
        Initialize transformation steps.
        Normalize, resize, and convert to tensor.
        '''
        # base transform
        transform_steps = None
        if transform_name == "base":
            transform_steps = self.base_transform(config)
        elif transform_name == "resize":
            transform_steps = self.resize_transform(config)
        elif transform_name == "resize_center_crop":
            transform_steps = self.resize_center_crop_transform(config)
        else:
            raise ValueError(f"Unknown transform: {transform_name}")
        

        transform_steps.append(transforms.ToTensor())

        # normalization
        default_normalization = transforms.Normalize(
            _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN,
            _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD)
        transform_steps.append(default_normalization)
        
        # compose
        self.transform = transforms.Compose(transform_steps)
    
    ## Helpers, based on code from WILDS
    def base_transform(self, config):
        transform_steps = []
        # center crop
        if (config['orig_res'] is not None and
            min(config['orig_res']) != max(config['orig_res'])):
            crop_size = min(config['orig_res'])
            transform_steps.append(transforms.CenterCrop(crop_size))
        # resize
        if config['target_res'] != config['orig_res']:
            transform_steps.append(transforms.Resize(config['target_res']))

        return transform_steps
    
    def resize_transform(self, config):
        """
        Resizes the image to a slightly larger square.
        """
        assert 'resize_scale' in config, "resize_scale required for resize transform."
        scaled_res = tuple(
            int(res * config['resize_scale']) 
            for res in config['orig_res'])
        
        return [transforms.Resize(scaled_res)]
    
    def resize_center_crop_transform(self, config):
        """
        Resizes the image to a slightly larger square then crops the center.
        """
        # resize
        transform_steps = self.resize_transform(config)
        # center crop
        transform_steps.append(transforms.CenterCrop(config['target_res']))

        return transform_steps
