'''
    Useful: transform.

    Supplement for torchvision.transforms.
'''

import numpy as np
import random
from PIL import Image


class AddGaussianNoise(object):
    """Add Gaussian Noise for range [0, 255] image."""
    def __init__(self, sigma, prob=1.):
        assert isinstance(sigma, float) and isinstance(prob, float)
        self.sigma = sigma
        self.prob = prob

    def __call__(self, img):
        if random.uniform(0, 1) < self.prob:
            clean_image = np.array(img).copy()
            noise_mask = np.random.randn(*clean_image.shape) * self.sigma
            result = clean_image + noise_mask
            result[result > 255] = 255
            result[result < 0] = 0
            return Image.fromarray(result.astype('uint8'))
        else:
            return img
