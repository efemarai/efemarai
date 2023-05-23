import builtins
from pydoc import locate

import numpy as np

from efemarai.metamorph.domain import Domain
from efemarai.metamorph.operators_registry import METAMORPHS
from efemarai.metamorph.vision.vision_metamorphs import ChooseImage


class DomainLoader:
    NUM_SAMPLES = 11

    @staticmethod
    def load(definition, dataset=None):
        domain = Domain()
        transformations = {
            transformation["name"]: transformation
            for transformation in definition["transformations"]
        }

        def traverse(transformation):
            for input in transformation["inputs"]:
                if input["output_from"] is None:
                    return

                traverse(transformations[input["output_from"]["transformation"]])

            operator, free_params = DomainLoader._reify_transformation(transformation)

            domain.add_transformation(
                id=transformation["name"],
                operator=operator,
                axes=free_params,
                inputs=transformation["inputs"],
                outputs=transformation["outputs"],
            )

        evaluate_sample = next(
            (
                transformation
                for transformation in transformations.values()
                if not transformation["outputs"]
            ),
            None,
        )

        traverse(evaluate_sample)

        return domain

    @staticmethod
    def _reify_transformation(transformation):
        if not transformation["inputs"]:
            return DomainLoader._reify_source(transformation)

        operator = transformation["operator"]

        free_params, fixed_params = DomainLoader._get_parameters(transformation)

        def operator_with_free_params_only(*args):
            kwargs = dict(zip(free_params.keys(), args))
            kwargs.update(fixed_params)
            return METAMORPHS[operator](**kwargs)

        return operator_with_free_params_only, free_params

    @staticmethod
    def _reify_source(transformation):
        if transformation["operator"] == "ChooseImage":
            return (ChooseImage, {"image": [-1]})

        raise ValueError("Transformation source cannot be established.")

    @staticmethod
    def _get_parameters(transformation):
        free = {}
        fixed = {}
        for axis in transformation["axes"]:
            try:
                if axis["fixed"]:
                    if axis["type"] in builtins.__dict__:
                        fixed[axis["name"]] = locate(axis["type"])(axis["value"])
                    else:
                        constructor = locate("efemarai.metamorph." + axis["type"])
                        fixed[axis["name"]] = constructor(*axis["value"])
                else:
                    free[axis["name"]] = DomainLoader._get_axis_values(axis)
            except Exception as e:
                print(
                    f"Error for loading transformation '{axis.name}' "
                    "with type '{axis.type}', value '{axis.value}'"
                )
                raise e
        return free, fixed

    @staticmethod
    def _get_axis_values(axis):
        if axis["type"] in ["float", "int"]:
            if axis.get("range"):
                low, high = axis["range"]
                return np.unique(
                    np.linspace(
                        low, high, DomainLoader.NUM_SAMPLES, dtype=locate(axis["type"])
                    )
                ).tolist()

            if axis.get("choices"):
                return [locate(axis["type"])(choice) for choice in axis["choices"]]

        if axis["type"] == "bool":
            return [False, True]

        if axis["type"] == "str":
            return [str(choice) for choice in axis["choices"]]

        raise ValueError(
            f"Only 'float', 'int', 'bool', 'str' types are supported for now."
            " Got {axis.type} for {axis.__dict__}"
        )
