from dataclasses import dataclass
from enum import Enum
from typing import Generic, Optional, Tuple, Type, TypeVar

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
    Text = "Text"


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
