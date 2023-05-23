from collections import Counter
from random import choice, sample


class Axis:
    """An axis of data variability.

    Attributes:
        name: Name of the axis describing the semantic property it represents.
        values: Sequence of all possible values along the axis.
    """

    def __init__(self, name, values):
        self.name = name
        self.values = values

    def __len__(self):
        return len(self.values)

    @property
    def min(self):
        """Get the min value along the axis."""
        return self.values[0]

    @property
    def max(self):
        """Get the max value along the axis."""
        return self.values[-1]

    def sample(self, n=None):
        """Get random value along the axis."""
        if n is None:
            return choice(self.values) if self.values else None

        # Sample without replacement
        if n <= len(self):
            return sample(self.values, n)

        # First sample without replacement, and then top up with replacement
        try:
            samples = self.sample(len(self))
            samples += [self.sample() for _ in range(n - len(self))]
        except Exception as e:
            from monitoring import log

            log.exception(
                f"Getting an issue sampling axis: {self.name} - {self.values}. n: {n} axis len: {len(self)}"
            )
            raise e

        return samples


class Input:
    def __init__(self, type, output_from, output_from_index):
        self.type = type
        self.output_from = output_from
        self.output_from_index = output_from_index


class Output:
    def __init__(self, type, input_to):
        self.type = type
        self.input_to = input_to


class Transformation:
    """A parameterized data transformation.

    Attributes:
        operator: Function called with parameter values to construct the operator
            that is to be applied to the data. The returned function should take a
            dict with targets as input and return a dict with transformed targets.
            A target may be an image, mask, masks, bboxes, keypoints, etc.
        param_names: Sequence with the names of parameters used by the transform.
    """

    def __init__(self, id, operator, param_names, inputs, outputs):
        assert isinstance(param_names, (tuple, list))
        self.id = id
        self.operator = operator
        self.param_names = param_names
        self.inputs = inputs
        self.outputs = outputs

    def __call__(self, params, *args):
        """Construct the transform operator using params and apply it to the targets.

        Args:
            params: Dictionary mapping a parameter name to its value.
            targets: Dictionary containing possible targets. Keys may include
                "image", "mask", "masks", "bboxes", "keypoints" or any other target
                supported by the transform operator.

        Returns:
            Dictionary with transformed targets.
        """
        assert all(name in params for name in self.param_names)

        # TODO: Figure out if 'valid' needs to be dealt with here
        # if not targets.get("valid", True):
        #     return targets

        param_values = tuple(params[name] for name in self.param_names)

        return self.operator(*param_values)(*args)


class Domain:
    """Operational domain defining the search space.

    A domain has a set of named axes that parameterize a set of transforms used
    for searching. The domain can also be viewed as a single transform that is the
    composition of all individual transforms.

    Attributes:
        axes: List of Axis objects.
        transformations: List of individual transfroms per axis/axes.
    """

    def __init__(self):
        self.axes = []
        self.transformations = {}
        self.sink = None

    @property
    def search_space(self):
        """Get the search space defined by the domain.

        Returns:
            A dict mapping axis names to the possible axis values.
        """
        return {axis.name: list(axis.values) for axis in self.axes}

    def add_transformation(self, id, operator, axes, inputs, outputs):
        """Insert new axes of variability.

        Args:
            axes: Dict mapping axis name to an iterable with all possible values.
            operator: Operator used by the transform.
        """
        self.transformations[id] = Transformation(
            id=id,
            operator=operator,
            param_names=tuple(axes.keys()),
            inputs=[
                Input(
                    type=input["type"],
                    output_from=self.transformations[
                        input["output_from"]["transformation"]
                    ],
                    output_from_index=input["output_from"]["index"],
                )
                for input in inputs
            ],
            outputs=[
                Output(
                    type=output["type"],
                    input_to=output["input_to"],
                )
                for output in outputs
            ],
        )

        if not outputs:
            self.sink = self.transformations[id]

        self.axes.extend([Axis(name, values) for name, values in axes.items()])

    def get_transformation(self, sink=None):
        """Get the overall transform the domain represents."""
        if sink is None:
            sink = self.sink

        def transform(params, *args):
            results = {}

            # Avoid storing all intermediate results/outputs, so keep a reference
            # count that is decreased every time the output is read. If the count
            # reaches 0 then the intermediate result/output is deleted.
            refs = Counter()

            def apply(transformation):
                inputs = []
                for _input in transformation.inputs:
                    output_from = _input.output_from
                    if output_from.id not in results:
                        results[output_from.id] = apply(output_from)
                        for output in output_from.outputs:
                            refs[output_from.id] += len(output.input_to)

                    inputs.append(results[output_from.id][_input.output_from_index])
                    refs[output_from.id] -= 1

                    if refs[output_from.id] == 0:
                        del results[output_from.id]
                        del refs[output_from.id]

                # At the beginning of the computation DAG
                if not inputs:
                    inputs = args

                return transformation(params, *inputs)

            return apply(sink)[0]

        return transform

    def sample(self, n=None):
        """Get a random sample from the search space of the domain.

        Returns:
            A dict mapping axis names to sampled values.
        """
        axes_samples = {axis.name: axis.sample(n) for axis in self.axes}

        if n is None:
            return axes_samples

        return [
            {axis.name: axes_samples[axis.name][i] for axis in self.axes}
            for i in range(n)
        ]

    def transform(self, *inputs, params=None):
        """Apply the transform of the domain to a set of targets

        Args:
            params: Dict containing parameter values at which the transform to be evaluated.
                If None a random sample will be generated.
            targets: List with targets to be transformed.

        Returns:
            A dict with transformed targets.
        """
        transformation = self.get_transformation()

        if params is not None:
            return transformation(params, *inputs)

        params = self.sample()
        return transformation(params, *inputs), params

    def generate(self, params=None):
        """Generate a set new targets.

        Often the search space will include an axis for choosing initial targets
        that are to be transformed. In that case the synthesized targets can
        be simply generated and not actually transformed.

        Args:
            params: Dict containing parameter values at which the transform to be
                evaluated. If None a random sample will be generated.

        Returns:
            A Datapoint of the generated sample.
        """
        transformation = self.get_transformation()

        if params is not None:
            return transformation(params)

        params = self.sample()

        return transformation(params), params
