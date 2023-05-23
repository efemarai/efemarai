from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Optional, Tuple, Type, TypeVar

import albumentations as A

from efemarai.metamorph.adaptors import apply_albumentation, apply_paste
from efemarai.metamorph.vision.operators.custom_operators import (
    GenericNoOp,
    MotionBlurDirectionLimit,
)

METAMORPHS = {}


@dataclass
class Box:
    tx: float
    ty: float
    bx: float
    by: float


@dataclass
class Color:
    r: int
    g: int
    b: int


T = TypeVar("T", bool, int, float, str, list, Color)


@dataclass
class OperatorParam(Generic[T]):
    name: str
    type: Type[T]
    choices: Tuple[T]
    range: Tuple[T]
    ordered: bool
    fixed: bool
    value: Optional[T]
    metadata_model: Optional[str]
    tf_model: Optional[str]


@dataclass
class OperatorInput:
    name: Optional[str]
    type: str


@dataclass
class OperatorOutput:
    name: Optional[str]
    type: str


class IOType(Enum):
    Datapoint = "Datapoint"


class Category(Enum):
    Undefined = "Undefined"
    Common = "Common"
    Geometric = "Geometric"
    Weather = "Weather"
    Color = "Color"
    Noise = "Noise"
    FaceWorks = "FaceWorks"
    MRI = "MRI"
    Image = "Image"


class Operator:
    """Create a Operator that holds the information for the functions in this module."""

    def __init__(self, operator):
        self.operator = operator
        self.name = self.operator.__name__
        self.category = Category.Undefined
        self.source = False
        self.sink = False
        self.merge = False
        self.params = []
        self.inputs = []
        self.outputs = []

    def add_param(
        self,
        name,
        type,
        choices,
        range,
        ordered,
        fixed,
        value,
        metadata_model,
        tf_model,
    ):
        self.params.append(
            OperatorParam(
                name,
                type,
                choices,
                range,
                ordered,
                fixed,
                value,
                metadata_model,
                tf_model,
            )
        )

    def add_input(self, name, type):
        self.inputs.append(OperatorInput(name, type.value))

    def add_output(self, name, type):
        self.outputs.append(OperatorOutput(name, type.value))

    def __call__(self, *args, **kwargs):
        return self.operator(*args, **kwargs)

    def __repr__(self):
        result = f"Operator: {self.operator} with params:"
        for parameter in self.params:
            result += "\n -"
            result += f" name: {parameter.name:12}"
            result += f" type: {parameter.type.__name__:5}"
            result += f" range: {str(parameter.range):10}"
            result += f" choices: {parameter.choices}"
        return result


def param(
    name,
    type,
    choices=None,
    range=None,
    ordered=False,
    fixed=False,
    value=None,
    metadata_model=None,
    tf_model=None,
):
    """Register a function as a transformation."""

    def decorator(func):
        if func.__name__ not in METAMORPHS:
            METAMORPHS[func.__name__] = Operator(func)

        METAMORPHS[func.__name__].add_param(
            name,
            type,
            choices,
            range,
            ordered,
            fixed,
            value,
            metadata_model,
            tf_model,
        )

        return func

    return decorator


def input(name=None, type=None):
    if type is None:
        type = IOType.Datapoint

    def decorator(func):
        if func.__name__ not in METAMORPHS:
            METAMORPHS[func.__name__] = Operator(func)

        METAMORPHS[func.__name__].add_input(name, type)

        return func

    return decorator


def output(name=None, type=None):
    if type is None:
        type = IOType.Datapoint

    def decorator(func):
        if func.__name__ not in METAMORPHS:
            METAMORPHS[func.__name__] = Operator(func)

        METAMORPHS[func.__name__].add_output(name, type)

        return func

    return decorator


def siso(input_name=None, input_type=None, output_name=None, output_type=None):
    if input_type is None:
        input_type = IOType.Datapoint

    if output_type is None:
        output_type = IOType.Datapoint

    def decorator(func):
        if func.__name__ not in METAMORPHS:
            METAMORPHS[func.__name__] = Operator(func)

        METAMORPHS[func.__name__].add_input(input_name, output_type)
        METAMORPHS[func.__name__].add_output(output_name, output_type)

        return func

    return decorator


def def_operator(category=None, source=False, sink=False, merge=False):
    """Add further info to an operator."""

    def decorator(func):
        if func.__name__ not in METAMORPHS:
            METAMORPHS[func.__name__] = Operator(func)

        if category is not None:
            METAMORPHS[func.__name__].category = category

        METAMORPHS[func.__name__].source = source
        METAMORPHS[func.__name__].sink = sink
        METAMORPHS[func.__name__].merge = merge

        # Stacked decorators are called in reverse order so fix by reversing again
        METAMORPHS[func.__name__].params.reverse()
        METAMORPHS[func.__name__].inputs.reverse()
        METAMORPHS[func.__name__].outputs.reverse()

        return func

    return decorator


@def_operator(category=Category.Noise)
@param("blur", int, range=(0, 20))
@siso()
@apply_albumentation(filter_instances=False)
def Blur(blur: int) -> Callable:
    """Applies Gaussian Blur to an image.

    Args:
        blur (int): The strength of the blur operator. Should be >= 3 to be active.

    Returns:
        Callable: A function that applies the blur operator to a target named variable `image`.
    """
    if blur < 3:
        return A.NoOp()

    assert blur >= 3, "blur operator size should be greater than 3."

    return A.Blur(blur_limit=(blur, blur), always_apply=True)


@def_operator(category=Category.Noise)
@param("median_blur_scale", int, range=(0, 15))
@siso()
@apply_albumentation(filter_instances=False)
def MedianBlur(median_blur_scale: int) -> Callable:
    """Applies Median Blur to an image.

    Args:
        median_blur_scale (int): The strength of the blur operator. blur = median_blur_scale * 2 + 1.

    Returns:
        Callable: A function that applies the median blur operator to a target named variable `image`.
    """
    if median_blur_scale == 0:
        return A.NoOp()

    assert (
        median_blur_scale >= 1
    ), "median_blur_scale operator size should be greater than 1."

    median_blur = median_blur_scale * 2 + 1

    return A.MedianBlur(blur_limit=(median_blur, median_blur), always_apply=True)


@def_operator(category=Category.Color)
@param("gamma", float, range=(0.5, 1.5))
@siso()
@apply_albumentation(filter_instances=False)
def Gamma(gamma: float) -> Callable:
    """Applies Gamma to an image.

    Args:
        gamma (float): The strength of the gamma operator.

    Returns:
        Callable: A function that applies the Gamma operator to a target named variable `image`.
    """
    assert gamma > 0, "gamma operator size should be greater than 0."

    return A.RandomGamma(
        gamma_limit=(int(gamma * 100), int(gamma * 100)), always_apply=True
    )


@def_operator(category=Category.Color)
@param("brightness", float, range=(-0.5, 0.5))
@param("contrast", float, range=(-0.5, 0.5))
@siso()
@apply_albumentation(filter_instances=False)
def BrightnessContrast(brightness: float, contrast: float = 0) -> Callable:
    """Applies Brightness and Contrast to an image.

    Args:
        brightness (float): The change in brightness.
        contrast (float): The change in contrast.

    Returns:
        Callable: A function that adds brightness or contrast operator to a target named
                  variable `image`.
    """
    return A.RandomBrightnessContrast(
        brightness_limit=(brightness, brightness),
        contrast_limit=(contrast, contrast),
        always_apply=True,
    )


@def_operator(category=Category.Color)
@param("intensity", float, range=(0, 1))
@param("light_direction", float, range=(-180, 180))
@siso()
def LightFlood(intensity: float, light_direction: float) -> Callable:
    """Applies Light Flood to an image.

    Args:
        intensity (float): Intensity of the incoming light.
        light_direction (float): Direction of the source of the light.

    Returns:
        Callable: A function that adds brightness or contrast operator to a target named
                  variable `image`.
    """
    return LightFloodOp(intensity=intensity, light_direction=light_direction)


@def_operator(category=Category.Noise)
@param("gauss_noise_var", float, range=(0, 10))
@param("gauss_noise_mean", float, range=(-10, 10), fixed=True, value=0)
@siso()
@apply_albumentation(filter_instances=False)
def GaussianNoise(gauss_noise_var: float, gauss_noise_mean: float = 0) -> Callable:
    """Applies Gaussian Noise to an image.

    Args:
        gauss_noise_var (float): The variance of the Gaussian noise operator. Should be > 0.
        gauss_noise_mean (float): The mean of the Gaussian noise operator.

    Returns:
        Callable: A function that applies the Gaussian noise operator to a target named variable `image`.
    """
    assert gauss_noise_var >= 0, "The gauss_noise_var should be > 0."

    return A.GaussNoise(
        var_limit=(gauss_noise_var, gauss_noise_var),
        mean=gauss_noise_mean,
        always_apply=True,
    )


@def_operator(category=Category.Color)
@param("hue", int, range=(-100, 100))
@param("sat", int, range=(-100, 100))
@param("val", int, range=(-100, 100))
@siso()
@apply_albumentation(filter_instances=False)
def ShiftHSV(hue: int, sat: int, val: int) -> Callable:
    """Shift the hue, saturation and value of an image.

    Args:
        hue (int): Change the hue by this value.
        sat (int): Change the saturation by this value.
        val (int): Change the value by this value.

    Returns:
        Callable: A function that shifts the HSV of target named variable `image`.
    """
    return A.HueSaturationValue(
        hue_shift_limit=(hue, hue),
        sat_shift_limit=(sat, sat),
        val_shift_limit=(val, val),
        always_apply=True,
    )


@def_operator(category=Category.Noise)
@param("motion_blur", int, range=(0, 20))
@param("motion_direction", float, range=(0, 1))
@siso()
@apply_albumentation(filter_instances=False)
def MotionBlur(motion_blur: int, motion_direction: float) -> Callable:
    """Adds motion blur to an image.

    Args:
        motion_blur (int): The blur amount. Should be >= 3 to be active.
        motion_direction (float): The direction of the motion. Should be in range [0..1].

    Returns:
        Callable: A function that adds motion blur to a target named variable `image`.
    """
    if motion_blur < 3:
        return A.NoOp()

    assert motion_blur >= 3, "motion_blur operator size should be greater than 3."
    assert 0 <= motion_direction <= 1, "motion_direction should be in range [0..1]"

    return MotionBlurDirectionLimit(
        blur_limit=(motion_blur, motion_blur),
        direction=motion_direction,
        always_apply=True,
    )


# NB: There is a stocastic effect here.
@def_operator(category=Category.Noise)
@param("max_factor", float, range=(1, 2))
@param("step_factor", float, range=(0, 0.05), fixed=True, value=0.025)
@siso()
@apply_albumentation(filter_instances=False)
def ZoomBlur(max_factor: int, step_factor: float) -> Callable:
    """Adds zoom blur to an image.

    Args:
        max_factor (float): Range for max factor for blurring
        step_factor (float): Step parameter for np.arange

    Returns:
        Callable: A function that adds zoom blur to a target named variable `image`.
    """
    if max_factor <= 1 or step_factor < 0.01:
        return A.NoOp()

    return A.ZoomBlur(
        max_factor=max_factor,
        step_factor=step_factor,
        always_apply=True,
    )


@def_operator(category=Category.Geometric)
@param("hflip", bool, choices=(False, True))
@siso()
@apply_albumentation()
def HorizontalFlip(hflip: bool) -> Callable:
    """Applies Horizontal Flip to an image.

    Args:
        hflip (bool): Should the image be flipped.

    Returns:
        Callable: A function that applies the horizontal flip operator to a target named variable `image`.
    """
    return A.HorizontalFlip(p=0, always_apply=hflip)


@def_operator(category=Category.Geometric)
@param("vflip", bool, choices=(False, True))
@siso()
@apply_albumentation()
def VerticalFlip(vflip: bool) -> Callable:
    """Applies Vertical Flip to an image.

    Args:
        vflip (bool): Should the image be flipped.

    Returns:
        Callable: A function that applies the Vertical flip operator to a target named variable `image`.
    """
    return A.VerticalFlip(p=0, always_apply=vflip)


@def_operator(category=Category.Geometric)
@param("affine_scale", float, range=(0.75, 1.25))
@param("shear_x", float, range=(-40, 40))
@param("shear_y", float, range=(-40, 40))
@param("affine_rotate", float, range=(-180, 180))
@param("translate_x", float, range=(-0.5, 0.5), fixed=True, value=0)
@param("translate_y", float, range=(-0.5, 0.5), fixed=True, value=0)
@param("interpolation", int, choices=(0, 1, 2, 3, 4), fixed=True, value=1)
@param("border_mode", int, choices=(0, 1, 2, 3, 4), fixed=True, value=1)
@param("infill_color", Color, range=((0, 255),) * 3, fixed=True, value=(0,) * 3)
@siso()
@apply_albumentation()
def Affine(
    affine_scale: float,
    translate_x: float = 0,
    translate_y: float = 0,
    affine_rotate: float = 0,
    shear_x: float = 0,
    shear_y: float = 0,
    interpolation: int = 1,
    border_mode: int = 1,
    infill_color: Tuple[int, int, int] = (0, 0, 0),
) -> Callable:
    """Applies Affine transforms to an image.

    Args:
        affine_scale (float): Scale factor of the image, where 1.0 denotes "no change" and 0.5 is zoomed out to 50% of the original size.
        translate_x (float): Translate image horizontally. 0 - no translation, 0.5 - translated by half the image.
        translate_y (float): Translate image vertically. 0 - no translation, 0.5 - translated by half the image.
        affine_rotate (float): Rotation in degrees around the centre of the image [-180..180].
        shear_x (float): Shear image horizontally by that many degrees (Usually in [-40, 40]).
        shear_y (float): Translate image vertically by that many degrees (Usually in [-40, 40]).
        interpolation (int): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_LINEAR (default - 1), cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.INTER_NEAREST.
        border_mode (int): Flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
        infill_color (int, int, int): The RGB color of the border created from the transformation.

    Returns:
        Callable: A function that applies Affine transforms to a target named variable `image`.
    """
    assert 0.5 <= affine_scale <= 1.5, "affine_scale should be in [0.5..1.5]."
    assert -0.5 <= translate_x <= 0.5, "translate_x should be in [-0.5..0.5]."
    assert -0.5 <= translate_y <= 0.5, "translate_y should be in [-0.5..0.5]."
    assert -180 <= affine_rotate <= 180, "affine_rotate should be in [-180..180]."
    assert -40 <= shear_x <= 40, "shear_x should be in [-40..40]."
    assert -40 <= shear_y <= 40, "shear_y should be in [-40..40]."
    assert interpolation in [0, 1, 2, 3, 4], "interpolation should be in [0..4]."
    assert border_mode in [0, 1, 2, 3, 4], "border_mode should be in [0..4]."

    if isinstance(infill_color, Color):
        infill_color = [infill_color.r, infill_color.g, infill_color.b]

    return A.Affine(
        scale=affine_scale,
        translate_percent={
            "x": (translate_x, translate_x),
            "y": (translate_y, translate_y),
        },
        rotate=(affine_rotate, affine_rotate),
        shear={"x": (shear_x, shear_x), "y": (shear_y, shear_y)},
        interpolation=interpolation,
        mode=border_mode,
        cval=infill_color,
        always_apply=True,
    )


# NB: There is a stocastic effect here
@def_operator(category=Category.Geometric)
@param("elastic_alpha", float, range=(0.0, 80))
@param("elastic_sigma", float, range=(-360, 360))
@param("elastic_alpha_affine", float, range=(0, 1))
@param("border_mode", int, choices=(0, 1, 2, 3, 4), fixed=True, value=0)
@param("interpolation", int, choices=(0, 1, 2, 3, 4), fixed=True, value=1)
@param("approximate", bool, choices=(True, False), fixed=True, value=True)
@param("same_dxdy", bool, choices=(True, False), fixed=True, value=True)
@siso()
@apply_albumentation()
def ElasticTransform(
    elastic_alpha: float,
    elastic_sigma: float = 0,
    elastic_alpha_affine: float = 0,
    interpolation: int = 1,
    border_mode: int = 0,
    approximate: bool = True,
    same_dxdy: bool = True,
) -> Callable:
    """Applies Elastic transforms to an image.

    Args:
        elastic_alpha (float): Alpha factor.
        elastic_sigma (float): Gaussian filter param.
        elastic_alpha_affine (float): Alpha affine factor.
        interpolation (int): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_LINEAR (default - 1), cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.INTER_NEAREST.
        border_mode (int): Flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
        approximate (bool): Whether to smooth displacement map with fixed kernel size. Disabling this option gives ~2X slowdown on large images.
        same_dxdy (bool): Whether to smooth displacement map with fixed kernel size. Disabling this option gives ~2X slowdown on large images.

    Returns:
        Callable: A function that applies Elastic transforms to a target named variable `image`.
    """
    assert 0 <= elastic_alpha <= 100, "elastic_alpha should be in [0.1..100]."
    assert -360 <= elastic_sigma <= 360, "elastic_sigma should be in [-360..360]."
    assert 0 <= elastic_alpha_affine <= 1, "elastic_alpha_affine should be in [0..1]."
    assert interpolation in [0, 1, 2, 3, 4], "interpolation should be in [0..4]."
    assert border_mode in [0, 1, 2, 3, 4], "border_mode should be in [0..4]."

    if elastic_alpha < 0.1:
        return A.NoOp()

    return A.ElasticTransform(
        alpha=elastic_alpha,
        sigma=elastic_sigma,
        alpha_affine=elastic_alpha_affine,
        interpolation=interpolation,
        border_mode=border_mode,
        approximate=approximate,
        same_dxdy=same_dxdy,
        always_apply=True,
    )


@def_operator(category=Category.Geometric)
@param("distort_limit", float, range=(-1, 1))
@param("distort_shift_limit", float, range=(-10, 10))
@param("border_mode", int, choices=(0, 1, 2, 3, 4), fixed=True, value=4)
@param("interpolation", int, choices=(0, 1, 2, 3, 4), fixed=True, value=1)
@siso()
@apply_albumentation()
def OpticalDistortion(
    distort_limit: float,
    distort_shift_limit: float = 0,
    interpolation: int = 1,
    border_mode: int = 4,
) -> Callable:
    """Applies OpticalDistortion transforms to an image.

    Args:
        distort_limit (float): Distortion factor
        distort_shift_limit (float): Shift factor
        interpolation (int): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_LINEAR (default - 1), cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.INTER_NEAREST.
        border_mode (int): Flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.

    Returns:
        Callable: A function that applies Optical Distortion transforms to a target named variable `image`.
    """
    assert -1 <= distort_limit <= 1, "distort_limit should be in [-1..1]."
    assert (
        -10 <= distort_shift_limit <= 10
    ), "distort_shift_limit should be in [-10..10]."
    assert interpolation in [0, 1, 2, 3, 4], "interpolation should be in [0..4]."
    assert border_mode in [0, 1, 2, 3, 4], "border_mode should be in [0..4]."

    return A.OpticalDistortion(
        distort_limit=(distort_limit, distort_limit),
        shift_limit=(distort_shift_limit, distort_shift_limit),
        interpolation=interpolation,
        border_mode=border_mode,
        always_apply=True,
    )


@def_operator(category=Category.Geometric)
@param("num_steps", int, range=(0, 10))
@param("distort_limit", float, range=(-0.3, 0.3))
@param("border_mode", int, choices=(0, 1, 2, 3, 4), fixed=True, value=4)
@param("border_color", Color, range=((0, 255),) * 3, fixed=True, value=(128,) * 3)
@param("interpolation", int, choices=(0, 1, 2, 3, 4), fixed=True, value=1)
@param("normalized", bool, choices=(False, True), fixed=True, value=False)
@siso()
@apply_albumentation()
def GridDistortion(
    num_steps: int,
    distort_limit: float,
    interpolation: int = 1,
    border_color: int = 255,
    border_mode: int = 4,
    normalized: bool = False,
) -> Callable:
    """Applies GridDistortion transforms to an image.

    Args:
        num_steps (int): count of grid cells on each side
        distort_limit (float): Distortion factor
        interpolation (int): Flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_LINEAR (default - 1), cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.INTER_NEAREST.
        border_color (Color): The color of the border when cv2.BORDER_CONSTANT
        border_mode (int): Flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
        normalzied (bool): Distortion to be normalized to do not go outside the image

    Returns:
        Callable: A function that applies GridDistortion transforms to a target named variable `image`.
    """
    assert interpolation in [0, 1, 2, 3, 4], "interpolation should be in [0..4]."
    assert border_mode in [0, 1, 2, 3, 4], "border_mode should be in [0..4]."

    if num_steps == 0:
        return A.NoOp()

    if isinstance(border_color, Color):
        border_color = [border_color.r, border_color.g, border_color.b]

    return A.GridDistortion(
        num_steps=num_steps,
        distort_limit=(distort_limit, distort_limit),
        interpolation=interpolation,
        border_mode=border_mode,
        normalized=normalized,
        value=border_color,
        always_apply=True,
    )


# NB: There is a stocastic effect here
@def_operator(category=Category.Geometric)
@param("perspective_scale", float, range=(0.0, 0.2))
@param("border_mode", int, choices=(0, 1, 2, 3, 4), fixed=True, value=0)
@siso()
@apply_albumentation()
def Perspective(
    perspective_scale: float,
    border_mode: int = 0,
) -> Callable:
    """Applies Perspective transforms to an image.

    Args:
        perspective_scale (float): Standard deviation of the normal distributions. These are used to sample the random distances of the subimage's corners from the full image's corners.
        border_mode (int): Flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.

    Returns:
        Callable: A function that applies Perspective transforms to a target named variable `image`.
    """
    if perspective_scale < 0.01:
        return A.NoOp()

    assert (
        0.01 <= perspective_scale <= 0.2
    ), "perspective_scale should be in [0.01..0.2]."
    assert border_mode in [0, 1, 2, 3, 4], "border_mode should be in [0..4]."

    return A.Perspective(
        scale=perspective_scale,
        keep_size=True,
        pad_mode=border_mode,
        always_apply=True,
    )


# NB: There is a stocastic effect here
@def_operator(category=Category.Geometric)
@param("scale", float, range=(0.0, 0.05))
@param("num_rows", int, range=(2, 5), fixed=True, value=3)
@param("num_cols", int, range=(2, 5), fixed=True, value=3)
@param("interpolation", int, choices=(0, 1, 2, 3, 4, 5), fixed=True, value=2)
@siso()
@apply_albumentation()
def PiecewiseAffine(
    scale: float,
    num_rows: int = 3,
    num_cols: int = 3,
    interpolation: int = 2,
) -> Callable:
    """Applies PiecewiseAffine transforms to an image.

    Args:
        scale (float): Each point on the regular grid is moved around via a normal distribution. This scale factor is equivalent to the normal distribution's sigma.
        nbrows (int): Number of rows of points that the regular grid should have. Must be at least 2. For large images, you might want to pick a higher value than 4.
        nbcols (int): Number of rows of points that the regular grid should have. Must be at least 2. For large images, you might want to pick a higher value than 4.
        interpolation (int): The order of interpolation. The order has to be in the range 0-5:
            - 0: Nearest-neighbor - 1: Bi-linear (default) - 2: Bi-quadratic - 3: Bi-cubic - 4: Bi-quartic - 5: Bi-quintic

    Returns:
        Callable: A function that applies PiecewiseAffine transforms to a target named variable `image`.
    """
    if scale < 0.01:
        return A.NoOp()

    assert num_rows >= 2, "num_rows must be at least 2"
    assert num_cols >= 2, "num_cols must be at least 2"

    return A.PiecewiseAffine(
        scale=scale,
        nb_rows=num_rows,
        nb_cols=num_cols,
        interpolation=interpolation,
        always_apply=True,
    )


@def_operator(category=Category.Color)
@param("sepia", bool, choices=(False, True))
@siso()
@apply_albumentation(filter_instances=False)
def ToSepia(sepia: bool) -> Callable:
    """Applies Sepia to an image. Sepia effect adds a warm brown tone to the image.

    Args:
        sepia (bool): Should the image have the sepia filter applied.

    Returns:
        Callable: A function that applies the Sepia operator to a target named variable `image`.
    """
    return A.ToSepia(p=0, always_apply=sepia)


@def_operator(category=Category.Color)
@param("gray", bool, choices=(False, True))
@siso()
@apply_albumentation(filter_instances=False)
def ToGray(gray: bool) -> Callable:
    """Applies grayscale to an image. If the mean pixel value for the resulting image is greater than 127, invert the resulting grayscale image.

    Args:
        gray (bool): Should the image have the grayscale filter applied.

    Returns:
        Callable: A function that applies the gray operator to a target named variable `image`.
    """
    return A.ToGray(p=0, always_apply=gray)


@def_operator(category=Category.Noise)
@param("iso_color_shift", float, range=(0, 1))
@param("iso_intensity", float, range=(0, 1), fixed=True, value=0.5)
@siso()
@apply_albumentation(filter_instances=False)
def ISONoise(iso_color_shift: float, iso_intensity: float = 0.25) -> Callable:
    """Applies ISO Noise to an image.

    Args:
        iso_color_shift (float): Variance range for color hue change. Measured as a fraction of 360 degree Hue angle in HLS colorspace in (0, 1).
        iso_intensity (float): Multiplicative factor that control strength of color and luminace noise.

    Returns:
        Callable: A function that applies ISO Noise operator to a target named variable `image`.
    """
    assert 0 <= iso_color_shift <= 1, "iso_color_shift should be [0, 1]."
    assert 0 <= iso_intensity <= 1, "iso_intensity should be [0, 1]."

    return A.ISONoise(
        color_shift=(iso_color_shift, iso_color_shift),
        intensity=(iso_intensity, iso_intensity),
        always_apply=True,
    )


@def_operator(category=Category.Noise)
@param("quality", int, range=(0, 100))
@siso()
@apply_albumentation(filter_instances=False)
def JPEGCompression(quality: int) -> Callable:
    """Applies Jpeg Compression to an image.

    Args:
        quality (int): Quality of the JPEG image in (0, 100).

    Returns:
        Callable: A function that applies Jpeg Compression operator to a target named variable `image`.
    """
    assert 0 <= quality <= 100, "quality operator should be [0, 100]."

    return A.JpegCompression(
        quality_lower=quality, quality_upper=quality, always_apply=True
    )


@def_operator(category=Category.Noise)
@param("image_quality", int, range=(1, 100))
@param("compression_type", str, choices=("jpg", "webp"), fixed=True, value="jpg")
@siso()
@apply_albumentation(filter_instances=False)
def ImageCompression(image_quality: int, compression_type: int = 0) -> Callable:
    """Applies Image Compression to an image.

    Args:
        image_quality (int): Quality of the compressed image in (1, 100).
        compression_type (int): Choose between JPEG (value: 0, default) and WebP (value: 1).

    Returns:
        Callable: A function that applies Image Compression operator to a target named variable `image`.
    """
    compression_type_list = ["jpg", "webp"]
    assert (
        compression_type in compression_type_list
    ), f"compression_type must be {compression_type_list}"
    compression_type = compression_type_list.index(compression_type)

    assert 0 < image_quality <= 100, "image_quality operator should be [1, 100]."

    return A.ImageCompression(
        quality_lower=image_quality,
        quality_upper=image_quality,
        compression_type=compression_type,
        always_apply=True,
    )


@def_operator(category=Category.Noise)
@param("clip_limit", float, range=(0.01, 3))
@param("tile_grid_size", int, range=(3, 15))
@siso()
@apply_albumentation(filter_instances=False)
def CLAHE(clip_limit: float, tile_grid_size: int = 0) -> Callable:
    """Apply Contrast Limited Adaptive Histogram Equalization to the image.

    Args:
        clip_limit (float): Threshold value for contrast limiting in (0.01, 3).
        tile_grid_size (int): threshold value for contrast limiting (size x size) in [3..15].

    Returns:
        Callable: A function that applies CLAHE operator to a target named variable `image`.
    """
    assert 0.01 <= clip_limit <= 3, "quality operator should be [0.01..3]."
    assert 3 <= tile_grid_size <= 15, "tile_grid_size should be in [3..15]."

    return A.CLAHE(
        clip_limit=(clip_limit, clip_limit),
        tile_grid_size=(tile_grid_size, tile_grid_size),
        always_apply=True,
    )


@def_operator(category=Category.Noise)
@param("equalize", bool, choices=(False, True))
@param("mode", str, choices=("cv", "pil"), fixed=True, value="cv")
@param("by_channels", bool, choices=(False, True), fixed=True, value=True)
@siso()
@apply_albumentation(filter_instances=False)
def Equalize(equalize: bool, mode: str = "cv", by_channels: bool = True) -> Callable:
    """Applies Equalization to an image historgram.

    Args:
        equalize (bool): Should the image be equalized.
        model (str): Use OpenCV or PIL image function.
        by_channels (bool): If True, use equalization by channels separately, else convert image to YCbCr representation and use equalization by Y channel.

    Returns:
        Callable: A function that applies Equalization operator to a target named variable `image`.
    """
    # assert 0 <= gray <= 1, "gray should be [0 or 1]."

    return A.Equalize(p=0, mode=mode, by_channels=by_channels, always_apply=equalize)


@def_operator(category=Category.Color)
@param("num_bits", int, range=(1, 8))
@siso()
@apply_albumentation(filter_instances=False)
def Posterize(num_bits: int) -> Callable:
    """Applies Posterize to an image.

    Args:
        num_bits (int): Reduce the number of bits for each color channel. Values in [1, 8].

    Returns:
        Callable: A function that applies Posterize operator to a target named variable `image`.
    """
    assert 1 <= num_bits <= 8, "num_bits should be [1, 8]."

    return A.Posterize(num_bits=num_bits, always_apply=True)


# NB: There is a stocastic effect here.
@def_operator(category=Category.Noise)
@param("scale", float, range=(0, 0.5))
@siso()
@apply_albumentation(filter_instances=False)
def ToneCurve(scale: float) -> Callable:
    """Applies ToneCurve to an image.
        Randomly change the relationship between bright and dark areas of the image by manipulating its tone curve.

    Args:
        scale (float): standard deviation of the normal distribution. Used to sample random
            distances to move two control points that modify the image's curve. Values should be in range [0, 1].

    Returns:
        Callable: A function that applies ToneCurve operator to a target named variable `image`.
    """
    assert 0 <= scale <= 0.5, "scale operator should be [0, 0.5]."

    return A.RandomToneCurve(scale=scale, always_apply=True)


@def_operator(category=Category.Noise)
@param("sharpen_alpha", float, range=(0, 1))
@param("sharpen_lightness", float, range=(0, 1), fixed=True, value=1)
@siso()
@apply_albumentation(filter_instances=False)
def Sharpen(sharpen_alpha: float, sharpen_lightness: float = 0.25) -> Callable:
    """Applies Sharpen to an image.

    Args:
        sharpen_alpha (float): Visibility of the sharpened image. At 0, only the original image is visible, at 1.0 only its sharpened version is visible.
        sharpen_lightness (float): Lightness of the sharpened image.

    Returns:
        Callable: A function that applies Sharpen operator to a target named variable `image`.
    """
    assert 0 <= sharpen_alpha <= 1, "sharpen_alpha should be [0, 1]."
    assert 0 <= sharpen_lightness <= 1, "sharpen_lightness should be [0, 1]."

    return A.Sharpen(
        alpha=(sharpen_alpha, sharpen_alpha),
        lightness=(sharpen_lightness, sharpen_lightness),
        always_apply=True,
    )


@def_operator(category=Category.Color)
@param("solarize_threshold", int, range=(0, 255))
@siso()
@apply_albumentation(filter_instances=False)
def Solarize(solarize_threshold: int) -> Callable:
    """Applies Solarize to an image.

    Args:
        solarize_threshold (int): Invert all pixel values above a threshold.

    Returns:
        Callable: A function that applies Solarize operator to a target named variable `image`.
    """
    assert 0 <= solarize_threshold <= 255, "solarize_threshold should be [0, 255]."

    return A.Solarize(threshold=solarize_threshold, always_apply=True)


@def_operator(category=Category.Image)
@param("occlusion_areas", int, range=(0, 20), value=2)
@param("max_height", float, range=(0, 1), fixed=True, value=0.1)
@param("max_width", float, range=(0, 1), fixed=True, value=0.1)
# @param("min_occlusion_areas", int, range=(0, 20), fixed=True, value=None)
# @param("min_height", float, range=(0, 1), fixed=True, value=None)
# @param("min_width", float, range=(0, 1), fixed=True, value=None)
@param("occlusion_color", Color, range=((0, 255),) * 3, fixed=True, value=(128,) * 3)
@siso()
@apply_albumentation(filter_instances=False)
def Occlusion(
    occlusion_areas=8,
    max_height=8,
    max_width=8,
    min_occlusion_areas=None,
    min_height=None,
    min_width=None,
    occlusion_color=0,
) -> Callable:
    """Occlude part of an image with a target color.

    Args:
        occlusion_areas (float): Maximum number of occlusions to apply.
        max_height (float): Height of the occluded area.
        max_width (float): Width of the occluded area.
        min_occlusion_areas (float): If specified, use the range between min and max occluded areas.
        min_height (float): If specified, range the height of the occluded area.
        min_width (float): If specified, range the width of the occluded area.
        occlusion_color (Color): The tripple to use to fill in the occluded parts.

    Returns:
        Callable: Occlude part of the image for image, mask, keypoints,
    """
    if occlusion_areas < 1:
        return A.NoOp()

    if isinstance(occlusion_color, Color):
        occlusion_color = [occlusion_color.r, occlusion_color.g, occlusion_color.b]

    return A.CoarseDropout(
        max_holes=occlusion_areas,
        max_height=max_height,
        max_width=max_width,
        min_holes=min_occlusion_areas,
        min_height=min_height,
        min_width=min_width,
        fill_value=occlusion_color,
        mask_fill_value=None,
        always_apply=True,
    )


@def_operator(category=Category.Weather)
@param("fog_coef", float, range=(0, 0.65))
@param("fog_alpha", float, range=(0, 1), fixed=True, value=0.25)
@siso()
@apply_albumentation(filter_instances=False)
def Fog(fog_coef: float, fog_alpha: float) -> Callable:
    """Applies Fog to an image.

    Args:
        fog_coef (float): The intensity of the fog. Should be in range [0..0.65].
        fog_alpha (float): The transparency of the fog cloud. Should be in range [0..1].

    Returns:
        Callable: A function that applies the fog operator to a target named variable `image`.
    """
    if fog_coef < 1e-2:
        return A.NoOp()

    assert 1e-2 <= fog_coef <= 0.65, "Fog coef needs to be in the [0..0.65] range"
    assert 0 <= fog_alpha <= 1, "Fog alpha needs to be in the [0..1] range"

    # fog_coef = 0 causes RandomFog to hang indefinitely despite of:
    # https://github.com/albumentations-team/albumentations/issues/361
    return A.RandomFog(
        fog_coef_lower=fog_coef,
        fog_coef_upper=fog_coef,
        alpha_coef=fog_alpha,
        always_apply=True,
    )


@def_operator(category=Category.Weather)
@param("snow_amount", float, range=(0, 1))
@param("snow_brightness", float, range=(0.01, 3), fixed=True, value=1.0)
@siso()
@apply_albumentation(filter_instances=False)
def Snow(snow_amount: float, snow_brightness: float) -> Callable:
    """Adds snow to an image.

    Args:
        snow_amount (float): The amount of the snow. Should be in range [0..1].
        snow_brightness (float): The brightness of the snow. Should be > 0.

    Returns:
        Callable: A function that adds snow to a target named variable `image`.
    """
    assert 0 <= snow_amount <= 1, "Snow amount must be in range [0..1]"
    assert snow_brightness > 0, "Snow brightness should be >0."

    return A.RandomSnow(
        snow_point_lower=snow_amount,
        snow_point_upper=snow_amount,
        brightness_coeff=snow_brightness,
        always_apply=True,
    )


@def_operator(category=Category.Weather)
@param("flare_angle", float, range=(0, 1))
@param("flare_cirles", int, range=(1, 8))
@param("flare_rad", int, range=(10, 500), fixed=True, value=200)
@param("flare_color", Color, range=((0, 255),) * 3, fixed=True, value=(255,) * 3)
@siso()
@apply_albumentation(filter_instances=False)
def SunFlare(
    flare_angle: float = 0.0,
    flare_cirles: int = 1,
    flare_rad: int = 200,
    flare_color: Tuple[int, int, int] = (255, 255, 255),
) -> Callable:
    """Applies SunFlare to an image.

    Args:
        flare_roi (float, float, float, float): The region in the image where the flare will appear.
            Top-left x,y, bottom right x, y. All values need to be within [0..1]
        flare_angle (float): The angle of the flare. Should be in range [0..1].
        flare_circles (int): The number of trailing flares. Should be > 0.
        flare_rad (int): The radius of the flare. Should be > 0, in pixels.
        flare_color (int, int, int): The RGB color of the flare.

    Returns:
        Callable: A function that applies the sun flare operator to a target named variable `image`.
    """
    flare_roi = Box(0, 0, 1, 1)
    if isinstance(flare_roi, Box):
        flare_roi = [flare_roi.tx, flare_roi.ty, flare_roi.bx, flare_roi.by]
    if isinstance(flare_color, Color):
        flare_color = [flare_color.r, flare_color.g, flare_color.b]

    for flare_r in flare_roi:
        assert 0 <= flare_r <= 1, "Flare_roi needs to be in the [0..1] range"
    assert 0 <= flare_angle <= 1, "Flare_angle needs to be in the [0..1] range"
    assert flare_cirles > 0, "Flare circles needs to be greater than 0."
    assert flare_rad > 0, "Flare radius needs to be greater than 0."

    flare_angle_lower = flare_angle - 0.01 if flare_angle != 0 else flare_angle
    flare_angle_upper = flare_angle + 0.01 if flare_angle != 1 else flare_angle

    return A.RandomSunFlare(
        flare_roi=flare_roi,
        angle_lower=flare_angle_lower,
        angle_upper=flare_angle_upper,
        num_flare_circles_lower=flare_cirles,
        num_flare_circles_upper=flare_cirles + 1,
        src_radius=flare_rad,
        src_color=flare_color,
        always_apply=True,
    )


@def_operator(category=Category.Weather)
@param("shadow_num", int, range=(0, 5))
@param("shadow_dimension", int, range=(2, 6), fixed=True, value=4)
@siso()
@apply_albumentation(filter_instances=False)
def Shadow(
    shadow_num: float = 1,
    shadow_dimension: int = 4,
) -> Callable:
    """Applies Shadow to an image.

    Args:
        shadow_roi (float, float, float, float): The region in the image where the shadow will appear.
            Top-left x,y, bottom right x, y. All values need to be within [0..1]
        shadows_num (int): The number of shadows. Should be in range [0..5].
        shadow_dimension (int): The number of edges in the shadow polygons. Should be in [2..6].

    Returns:
        Callable: A function that applies the shadow operator to a target named variable `image`.
    """
    shadow_roi = Box(0, 0, 1, 1)
    if isinstance(shadow_roi, Box):
        shadow_roi = [shadow_roi.tx, shadow_roi.ty, shadow_roi.bx, shadow_roi.by]

    for shadow_r in shadow_roi:
        assert 0 <= shadow_r <= 1, "shadow_roi needs to be in the [0..1] range"
    assert 0 <= shadow_num <= 5, "shadows_num needs to be in the [0..5] range"
    assert (
        2 <= shadow_dimension <= 6
    ), "shadow_dimension needs to be in the [2..6] range"

    return A.RandomShadow(
        shadow_roi=shadow_roi,
        num_shadows_lower=shadow_num,
        num_shadows_upper=shadow_num,
        shadow_dimension=shadow_dimension,
        always_apply=True,
    )


@def_operator(category=Category.Weather)
@param("intensity", float, range=(0, 1))
@param("mean", float, range=(0, 1), fixed=True, value=0.68)
@param("std", float, range=(0.05, 1))
@param("gauss_sigma", float, range=(0, 1))
@param("cutout_threshold", float, range=(0, 1), fixed=True, value=0.5)
@param("mode", str, choices=("rain", "mud"), fixed=True, value="rain")
@siso()
@apply_albumentation(filter_instances=False)
def Spatter(
    intensity: float,
    mean: float,
    std: float,
    gauss_sigma: float,
    cutout_threshold: float,
    mode: str,
) -> Callable:
    """Applies Spatter to an image.

    Args:
        intensity (float): Intensity of corruption
        mean (float): Mean value of normal distribution for generating liquid layer
        std (float): Standard deviation value of normal distribution for generating liquid layer
        gauss_sigma (float): Sigma value for gaussian filtering of liquid layer
        cutout_threshold (float): Threshold for filtering liqued layer (determines number of drops)
        mode (str): Type of corruption. Currently, supported options are 'rain' and 'mud'
    Returns:
        Callable: A function that applies the Spatter operator to a target named variable `image`.
    """
    if intensity == 0:
        return A.NoOp()

    return A.Spatter(
        intensity=intensity,
        mean=mean,
        std=std,
        gauss_sigma=gauss_sigma,
        cutout_threshold=cutout_threshold,
        mode=mode,
        always_apply=True,
    )


@def_operator(category=Category.Common, source=True)
@param("dataset", str, fixed=True, value="")
@param("classes", list, fixed=True, value=[])
@output()
def ChooseImage(datapoint_id) -> Callable:
    """Chooses an image from a dataset."""
    return GenericNoOp()


@def_operator(category=Category.Common, sink=True)
@input()
def EvaluateSample():
    """Operator used to denote the output of the sample generation process."""
    return lambda datapoint: (datapoint,)


if __name__ == "__main__":
    print(f"Avaliable operators: {METAMORPHS.keys()}")
