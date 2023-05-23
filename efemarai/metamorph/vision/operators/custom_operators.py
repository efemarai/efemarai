import random
from math import cos, pi, sin

import albumentations as A
import cv2
import numpy as np
from skimage.transform import PiecewiseAffineTransform


class GenericNoOp:
    """Creates a generic no operation that doesn't depend on any pre tf."""

    def __call__(self, *args):
        return args


class MotionBlurDirectionLimit(A.Blur):
    """Apply motion blur to the input image using a random-sized kernel.

    Args:
        blur_limit (int): maximum kernel size for blurring the input image.
            Should be in range [3, inf). Default range: (3, 9).
        direction (float): angle of the motion blur in degrees. Default: 0.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, blur_limit=(3, 9), direction=0, always_apply=False, p=0.5):
        super().__init__(blur_limit, always_apply, p)
        self.direction = direction

    @staticmethod
    def apply(img, kernel=None, **params):
        return A.functional.convolve(img, kernel=kernel)

    def get_params(self):
        ksize = random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2))
        if ksize <= 2:
            raise ValueError(f"ksize must be > 2. Got: {ksize}")
        kernel = np.zeros((ksize, ksize), dtype=np.uint8)

        # Choose direction
        angle = self.direction * pi / 180.0
        b = cos(angle)
        a = sin(angle)
        x0, y0 = ksize // 2, ksize // 2
        pt0 = (int(x0), int(y0))
        pt1 = (int(x0 + a * 2 * ksize), int(y0 - b * 2 * ksize))
        cv2.line(kernel, pt0, pt1, 1, thickness=1)

        # Normalize kernel
        kernel = kernel.astype(np.float32) / np.sum(kernel)
        return {"kernel": kernel}


class FastPiecewiseTransform(PiecewiseAffineTransform):
    def __call__(self, coords):
        coords = np.asarray(coords)

        simplex = self._tesselation.find_simplex(coords)

        affines = np.array(
            [self.affines[i].params for i, _ in enumerate(self._tesselation.simplices)]
        )[simplex]

        pts = np.c_[coords, np.ones((coords.shape[0], 1))]

        result = np.einsum("ij,ikj->ik", pts, affines)
        result[simplex == -1, :] = -1

        return result
